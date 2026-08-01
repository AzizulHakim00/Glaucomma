        # seg_prob at image resolution: B,2,H,W; reduce to feature map.
        B,C,Hf,Wf = feat.shape
        p = F.interpolate(seg_prob, size=(Hf,Wf), mode="bilinear", align_corners=False)
        disc = p[:,0:1].clamp(1e-5,1); cup = (p[:,1:2] * disc).clamp(0,1)
        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,Hf,device=feat.device), torch.linspace(-1,1,Wf,device=feat.device), indexing="ij"
        )
        xx = xx[None,None]; yy = yy[None,None]
        mass = disc.sum((2,3), keepdim=True).clamp_min(1e-4)
        cx = (disc*xx).sum((2,3), keepdim=True)/mass
        cy = (disc*yy).sum((2,3), keepdim=True)/mass
        ang = (torch.atan2(yy-cy, xx-cx) + 2*math.pi) % (2*math.pi)
        sector_id = torch.floor(ang/(2*math.pi/self.num_sectors)).long().clamp(0,self.num_sectors-1)
        nodes=[]; rim_ratios=[]
        rim = (disc-cup).clamp(0,1)
        for i in range(self.num_sectors):
            sm = (sector_id==i).float()
            w = disc*sm
            denom = w.sum((2,3)).clamp_min(1e-4)
            vf = (feat*w).sum((2,3))/denom
            d_area=(disc*sm).mean((2,3)); c_area=(cup*sm).mean((2,3)); r_area=(rim*sm).mean((2,3))
            local_cdr=(c_area/d_area.clamp_min(1e-5)).clamp(0,1)
            conf=(disc*sm).amax((2,3))
            geom=torch.stack([d_area.squeeze(1),c_area.squeeze(1),local_cdr.squeeze(1),conf.squeeze(1)],dim=1)
            nodes.append(torch.cat([vf,geom],dim=1)); rim_ratios.append((r_area/d_area.clamp_min(1e-5)).squeeze(1))
        nodes=torch.stack(nodes,dim=1)
        gp=F.adaptive_avg_pool2d(feat,1).flatten(1)
        d_area=disc.mean((2,3)); c_area=cup.mean((2,3)); cdar=(c_area/d_area.clamp_min(1e-5)).clamp(0,1)
        # differentiable approximate vertical CDR using soft vertical projections and variance.
        yd=(disc*yy).sum((2,3))/mass.squeeze(-1).squeeze(-1)
        cup_mass=cup.sum((2,3),keepdim=True).clamp_min(1e-4)
        yc=(cup*yy).sum((2,3))/cup_mass.squeeze(-1).squeeze(-1)
        sd=torch.sqrt(((disc*(yy-yd[:,:,None,None])**2).sum((2,3))/mass.squeeze(-1).squeeze(-1)).clamp_min(1e-6))
        sc=torch.sqrt(((cup*(yy-yc[:,:,None,None])**2).sum((2,3))/cup_mass.squeeze(-1).squeeze(-1)).clamp_min(1e-6))
        vcdr=(sc/sd.clamp_min(1e-5)).clamp(0,1)
        global_geom=torch.cat([d_area,c_area,cdar,vcdr],dim=1)
        global_node=torch.cat([gp,global_geom],dim=1).unsqueeze(1)
        nodes=torch.cat([nodes,global_node],dim=1)
        struct=torch.cat([global_geom,torch.stack(rim_ratios,dim=1)],dim=1)
        local=(feat*disc).sum((2,3))/disc.sum((2,3)).clamp_min(1e-4)
        return nodes,struct,local,vcdr.squeeze(1)

    def forward(self, x, grl_lambda=0.0):
        feats=self.backbone(x); f=feats[-1]
        seg_small=self.seg_head(f)
        seg=F.interpolate(seg_small,size=x.shape[-2:],mode="bilinear",align_corners=False)
        prob=torch.sigmoid(seg)
        global_raw=F.adaptive_avg_pool2d(f,1).flatten(1)
        nodes,struct,local_raw,vcdr_mask=self._soft_geometry(prob,f)
        z,a1=self.gat1(nodes,self.adjacency); z,a2=self.gat2(z,self.adjacency)
        graph_raw=z.mean(1)
        fg=self.global_proj(global_raw); fl=self.local_proj(local_raw); fr=self.graph_proj(graph_raw); fs=self.struct_proj(struct)
        q=torch.sigmoid(self.quality_head(global_raw))
        gates=torch.softmax(self.gate(torch.cat([fg,fl,fr,fs,q],dim=1)),dim=1)
        stack=torch.stack([fg,fl,fr,fs],dim=1)
        fused=(stack*gates.unsqueeze(-1)).sum(1)
        cls=self.classifier(fused).squeeze(1)
        vcdr_reg=self.vcdr_head(fused).squeeze(1)
        domain=self.domain_head(grad_reverse(fused,grl_lambda))
        return {"logit":cls,"seg":seg,"vcdr_reg":vcdr_reg,"vcdr_mask":vcdr_mask,"domain":domain,
                "fused":fused,"gates":gates,"quality":q.squeeze(1),"graph_attention":a2.mean(1)}

# ---------------------------- 9. LOSSES + METRICS ----------------------------
def binary_focal_with_logits(logits, targets, gamma=2.0, alpha=None):
    bce=F.binary_cross_entropy_with_logits(logits,targets,reduction="none")
    p=torch.sigmoid(logits); pt=p*targets+(1-p)*(1-targets)
    loss=(1-pt).pow(gamma)*bce
    if alpha is not None: loss=loss*(alpha*targets+(1-alpha)*(1-targets))
    return loss.mean()

def dice_loss(logits,target,valid):
    p=torch.sigmoid(logits)
    inter=(p*target).sum((2,3)); den=(p+target).sum((2,3))
    loss=1-(2*inter+1)/(den+1)
    return (loss.mean(1)*valid).sum()/valid.sum().clamp_min(1)

def segmentation_loss(seg,disc,cup,valid):
    target=torch.cat([disc,cup],1)
    bce=F.binary_cross_entropy_with_logits(seg,target,reduction="none").mean((1,2,3))
    bce=(bce*valid).sum()/valid.sum().clamp_min(1)
    return bce+dice_loss(seg,target,valid)

def prototype_loss(features,labels,domains):
    terms=[]
    for c in [0,1]:
        protos=[]
        for d in domains.unique():
            m=(labels.long()==c)&(domains==d)
            if m.sum()>0: protos.append(features[m].mean(0))
        for i in range(len(protos)):
            for j in range(i+1,len(protos)): terms.append(F.mse_loss(protos[i],protos[j]))
    return torch.stack(terms).mean() if terms else features.sum()*0

def expected_calibration_error(y,p,bins=15):
    edges=np.linspace(0,1,bins+1); ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>lo)&(p<=hi)
        if m.any(): ece+=m.mean()*abs(y[m].mean()-p[m].mean())
    return float(ece)

def safe_auc(y,p):
    return roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan

def classification_metrics(y,p,threshold=.5):
    y=np.asarray(y).astype(int); p=np.asarray(p); pred=(p>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {
        "auroc":safe_auc(y,p), "auprc":average_precision_score(y,p) if len(np.unique(y))>1 else np.nan,
        "accuracy":accuracy_score(y,pred), "balanced_accuracy":balanced_accuracy_score(y,pred),
        "sensitivity":tp/max(1,tp+fn), "specificity":tn/max(1,tn+fp),
        "precision":precision_score(y,pred,zero_division=0), "f1":f1_score(y,pred,zero_division=0),
        "mcc":matthews_corrcoef(y,pred) if len(np.unique(pred))>1 else 0.0,
        "brier":brier_score_loss(y,p), "nll":log_loss(y,np.c_[1-p,p],labels=[0,1]),
        "ece":expected_calibration_error(y,p), "threshold":threshold,
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)
    }

def dice_metric(pred,target,valid):
    pred=(pred>.5).float(); inter=(pred*target).sum((2,3)); den=(pred+target).sum((2,3))
    d=(2*inter+1)/(den+1)
    valid=valid[:,None]
    return ((d*valid).sum(0)/valid.sum().clamp_min(1)).detach().cpu().numpy()

# ---------------------------- 10. SPLITS + LOADERS ----------------------------
def source_stratified_split(df, val_fraction, seed):
    tr_parts=[]; va_parts=[]
    for src,g in df.groupby("source"):
        if g["label"].nunique()<2 or len(g)<10:
            n=max(1,int(len(g)*val_fraction)); va=g.sample(n=n,random_state=seed); tr=g.drop(va.index)
        else:
            splitter=StratifiedShuffleSplit(n_splits=1,test_size=val_fraction,random_state=seed)
            ti,vi=next(splitter.split(g,g["label"])); tr=g.iloc[ti]; va=g.iloc[vi]
        tr_parts.append(tr); va_parts.append(va)
    return pd.concat(tr_parts).reset_index(drop=True),pd.concat(va_parts).reset_index(drop=True)

def make_loader(df,train):
    ds=GlaucomaDataset(df,train=train)
    if train:
        # Joint source/class weighting.
        freq=df.groupby(["source","label"]).size().to_dict()

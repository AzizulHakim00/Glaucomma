        weights=[1.0/freq[(r.source,r.label)] for r in df.itertuples()]
        sampler=WeightedRandomSampler(weights,len(weights),replacement=True)
        return DataLoader(ds,batch_size=CFG["batch_size"],sampler=sampler,num_workers=CFG["num_workers"],pin_memory=True,drop_last=len(ds)>=CFG["batch_size"])
    return DataLoader(ds,batch_size=CFG["batch_size"],shuffle=False,num_workers=CFG["num_workers"],pin_memory=True)

# ---------------------------- 11. TRAIN/EVAL ----------------------------
def batch_to_device(batch):
    for k in ["image","disc","cup","label","domain","mask_valid","vcdr"]: batch[k]=batch[k].to(DEVICE,non_blocking=True)
    return batch

def run_epoch(model,loader,optimizer=None,scaler=None,epoch=0):
    train=optimizer is not None; model.train(train)
    all_y=[];all_p=[]; losses=[];dices=[]
    optimizer and optimizer.zero_grad(set_to_none=True)
    for bi,b in enumerate(loader):
        b=batch_to_device(b)
        progress=(epoch+bi/max(1,len(loader)))/max(1,CFG["epochs"])
        grl=2/(1+math.exp(-10*progress))-1 if train else 0
        with torch.autocast(device_type=DEVICE.type,enabled=CFG["mixed_precision"] and DEVICE.type=="cuda"):
            o=model(b["image"],grl)
            alpha=float((1-b["label"].mean()).clamp(.25,.75))
            l_cls=binary_focal_with_logits(o["logit"],b["label"],CFG["focal_gamma"],alpha)
            l_seg=segmentation_loss(o["seg"],b["disc"],b["cup"],b["mask_valid"])
            l_vcdr=((F.smooth_l1_loss(o["vcdr_reg"],b["vcdr"],reduction="none")*b["mask_valid"]).sum()/b["mask_valid"].sum().clamp_min(1))
            l_cons=F.smooth_l1_loss(o["vcdr_reg"],o["vcdr_mask"].detach())
            l_domain=F.cross_entropy(o["domain"],b["domain"])
            l_proto=prototype_loss(o["fused"],b["label"],b["domain"])
            loss=l_cls+CFG["lambda_seg"]*l_seg+CFG["lambda_vcdr"]*l_vcdr+CFG["lambda_cons"]*l_cons+CFG["lambda_domain"]*l_domain+CFG["lambda_proto"]*l_proto
            loss=loss/CFG["grad_accum"]
        if train:
            scaler.scale(loss).backward()
            if (bi+1)%CFG["grad_accum"]==0 or bi+1==len(loader):
                scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),CFG["max_grad_norm"])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu())*CFG["grad_accum"])
        p=torch.sigmoid(o["logit"]).detach().cpu().numpy(); y=b["label"].detach().cpu().numpy()
        all_y.extend(y);all_p.extend(p)
        d=dice_metric(torch.sigmoid(o["seg"]),torch.cat([b["disc"],b["cup"]],1),b["mask_valid"]);dices.append(d)
        if CFG["fast_dev_run"] and bi>=2: break
    m=classification_metrics(np.array(all_y),np.array(all_p)); dd=np.nanmean(np.stack(dices),axis=0)
    m.update({"loss":np.mean(losses),"dice_disc":float(dd[0]),"dice_cup":float(dd[1])})
    return m

@torch.no_grad()
def predict(model,loader):
    model.eval(); rows=[];dices=[]
    for b in loader:
        b=batch_to_device(b);o=model(b["image"],0)
        p=torch.sigmoid(o["logit"]).cpu().numpy(); seg=torch.sigmoid(o["seg"])
        d=dice_metric(seg,torch.cat([b["disc"],b["cup"]],1),b["mask_valid"]);dices.append(d)
        for i in range(len(p)):
            sector_imp = o["graph_attention"][i, :CFG["num_sectors"], :CFG["num_sectors"]].mean(0).cpu().numpy()
            rows.append({"path":b["path"][i],"source":b["source_name"][i],"label":int(b["label"][i].cpu()),
                         "prob_raw":float(p[i]),"vcdr_pred":float(o["vcdr_reg"][i].cpu()),"vcdr_mask":float(o["vcdr_mask"][i].cpu()),
                         "quality":float(o["quality"][i].cpu()),
                         **{f"gate_{n}":float(o["gates"][i,j].cpu()) for j,n in enumerate(["global","local","graph","struct"])},
                         **{f"sector_{j+1:02d}":float(sector_imp[j]) for j in range(CFG["num_sectors"])}})
        if CFG["fast_dev_run"] and len(rows)>=12: break
    dd=np.nanmean(np.stack(dices),axis=0)
    return pd.DataFrame(rows),{"dice_disc":float(dd[0]),"dice_cup":float(dd[1])}

# ---------------------------- 12. CALIBRATION ----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self): super().__init__(); self.log_t=nn.Parameter(torch.zeros(1))
    @property
    def temperature(self): return self.log_t.exp().clamp(.05,10)
    def forward(self,logits): return logits/self.temperature

def fit_temperature(y,p):
    eps=1e-6; p=np.clip(p,eps,1-eps); logits=torch.tensor(np.log(p/(1-p)),dtype=torch.float32)
    targets=torch.tensor(y,dtype=torch.float32); model=TemperatureScaler()
    opt=torch.optim.LBFGS(model.parameters(),lr=.05,max_iter=80,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad();loss=F.binary_cross_entropy_with_logits(model(logits),targets);loss.backward();return loss
    opt.step(closure);return float(model.temperature.detach())

def fit_robust_temperature(y, p, domains, steps=250):
    """Minimizes a smooth approximation of worst-source validation NLL."""
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    logits=torch.tensor(np.log(p/(1-p)),dtype=torch.float32)
    targets=torch.tensor(np.asarray(y),dtype=torch.float32)
    domains=np.asarray(domains)
    log_t=torch.nn.Parameter(torch.zeros(1))
    opt=torch.optim.Adam([log_t],lr=.03)
    unique=list(pd.unique(domains))
    for _ in range(steps):
        T=log_t.exp().clamp(.05,10); losses=[]
        for d in unique:
            idx=torch.tensor(domains==d)
            losses.append(F.binary_cross_entropy_with_logits(logits[idx]/T,targets[idx]))
        loss=.10*torch.logsumexp(torch.stack(losses)/.10,dim=0)
        opt.zero_grad();loss.backward();opt.step()
    return float(log_t.exp().clamp(.05,10).detach())

def apply_temperature(p,T):
    p=np.clip(np.asarray(p),1e-6,1-1e-6);z=np.log(p/(1-p))/T;return 1/(1+np.exp(-z))

def optimal_threshold(y,p):
    fpr,tpr,thr=roc_curve(y,p);j=tpr-fpr;return float(thr[np.argmax(j)])

# ---------------------------- 13. FIGURES ----------------------------
def save_evaluation_figures(pred,metrics,fold):
    out_rel=f"folds/{fold}/figures"; STORE.dirs(out_rel)
    y=pred.label.values;p=pred.prob_calibrated.values;pr=(p>=metrics["threshold"]).astype(int)
    fpr,tpr,_=roc_curve(y,p);prec,rec,_=precision_recall_curve(y,p)
    fig=plt.figure(figsize=(14,4))
    ax=fig.add_subplot(1,3,1);ax.plot(fpr,tpr,label=f"AUC={metrics['auroc']:.3f}");ax.plot([0,1],[0,1],'--');ax.set_title("ROC");ax.set_xlabel("FPR");ax.set_ylabel("TPR");ax.legend();ax.grid(alpha=.2)
    ax=fig.add_subplot(1,3,2);ax.plot(rec,prec,label=f"AP={metrics['auprc']:.3f}");ax.set_title("Precision–Recall");ax.set_xlabel("Recall");ax.set_ylabel("Precision");ax.legend();ax.grid(alpha=.2)
    cm=confusion_matrix(y,pr,labels=[0,1]);ax=fig.add_subplot(1,3,3);im=ax.imshow(cm);ax.set_title("Confusion matrix");ax.set_xticks([0,1]);ax.set_yticks([0,1]);ax.set_xlabel("Predicted");ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):ax.text(j,i,cm[i,j],ha="center",va="center")
    plt.tight_layout();STORE.save_figure(fig,f"{out_rel}/discrimination.png");display(fig);plt.close(fig)
    frac,mean=calibration_curve(y,p,n_bins=10,strategy="quantile")
    fig=plt.figure(figsize=(6,5));plt.plot(mean,frac,"o-",label=f"ECE={metrics['ece']:.3f}");plt.plot([0,1],[0,1],'--');plt.xlabel("Mean predicted probability");plt.ylabel("Observed fraction");plt.title(f"Calibration — held-out {fold}");plt.legend();plt.grid(alpha=.2);STORE.save_figure(fig,f"{out_rel}/calibration.png");plt.close(fig)

def save_visual_examples(model,df,fold,n=6):
    if len(df)==0:return
    sample=df.sample(min(n,len(df)),random_state=CFG["seed"])
    ds=GlaucomaDataset(sample,train=False);loader=DataLoader(ds,batch_size=1,shuffle=False)
    fig=plt.figure(figsize=(18,4*len(sample)))
    mean=np.array([.485,.456,.406]);std=np.array([.229,.224,.225])
    model.eval()
    for r,b in enumerate(loader):
        b=batch_to_device(b)
        x=b["image"].detach().clone().requires_grad_(True)
        o=model(x,0); prob=torch.sigmoid(o["logit"])[0]
        grad=torch.autograd.grad(o["logit"][0],x,retain_graph=False,create_graph=False)[0]
        sal=grad[0].abs().mean(0).detach().cpu().numpy();sal=(sal-sal.min())/(sal.max()-sal.min()+1e-8)
        seg=torch.sigmoid(o["seg"])[0].detach().cpu().numpy()
        img=x[0].detach().cpu().permute(1,2,0).numpy()*std+mean;img=np.clip(img,0,1)
        gt=np.zeros((*img.shape[:2],3));gt[...,1]=b["disc"][0,0].cpu();gt[...,0]=b["cup"][0,0].cpu()
        pdm=np.zeros_like(gt);pdm[...,1]=seg[0];pdm[...,0]=seg[1]
        att=o["graph_attention"][0,:CFG["num_sectors"],:CFG["num_sectors"]].mean(0).detach().cpu().numpy()
        ax=fig.add_subplot(len(sample),5,r*5+1);ax.imshow(img);ax.set_title(f"True={int(b['label'][0])}, P={float(prob):.3f}");ax.axis('off')
        ax=fig.add_subplot(len(sample),5,r*5+2);ax.imshow(img);ax.imshow(gt,alpha=.45);ax.set_title("Ground truth disc/cup");ax.axis('off')
        ax=fig.add_subplot(len(sample),5,r*5+3);ax.imshow(img);ax.imshow(pdm,alpha=.45);ax.set_title("Predicted disc/cup");ax.axis('off')
        ax=fig.add_subplot(len(sample),5,r*5+4);ax.imshow(img);ax.imshow(sal,cmap="jet",alpha=.45);ax.set_title("Input-gradient XAI");ax.axis('off')
        ax=fig.add_subplot(len(sample),5,r*5+5);ax.bar(np.arange(1,len(att)+1),att);ax.set_title("Clock-sector attention");ax.set_xlabel("Sector")
    plt.tight_layout();STORE.save_figure(fig,f"folds/{fold}/figures/qualitative_xai_examples.png",dpi=160);display(fig);plt.close(fig)


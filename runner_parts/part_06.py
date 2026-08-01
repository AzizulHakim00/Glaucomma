# ---------------------------- 14. CHECKPOINT ----------------------------
def atomic_torch_save(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");torch.save(obj,tmp);tmp.replace(path)

def save_checkpoint(obj,rel):
    lp=LOCAL_RUN/rel;atomic_torch_save(obj,lp);STORE.mirror(lp,rel);return lp

def find_existing(rel):
    lp=LOCAL_RUN/rel;dp=DRIVE_RUN/rel
    if lp.exists():return lp
    if dp.exists():lp.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dp,lp);return lp
    return None

# ---------------------------- 14B. SOURCE-ONLY OPTUNA TUNING ----------------------------
def tune_fold_hyperparameters(train_df, val_df, target_source):
    if not CFG["run_optuna"]: return {}
    import optuna
    base={k:CFG[k] for k in ["lr","weight_decay","dropout","lambda_proto","lambda_cons","lambda_domain"]}
    train_loader=make_loader(train_df,True); val_loader=make_loader(val_df,False)
    def objective(trial):
        CFG["lr"]=trial.suggest_float("lr",5e-5,4e-4,log=True)
        CFG["weight_decay"]=trial.suggest_float("weight_decay",1e-6,5e-3,log=True)
        CFG["dropout"]=trial.suggest_float("dropout",.10,.45)
        CFG["lambda_proto"]=trial.suggest_float("lambda_proto",.01,.25,log=True)
        CFG["lambda_cons"]=trial.suggest_float("lambda_cons",.05,.40)
        CFG["lambda_domain"]=trial.suggest_float("lambda_domain",.01,.20,log=True)
        model=RimGraphDG(num_domains=train_df.source.nunique()).to(DEVICE)
        bp=list(model.backbone.parameters()); ids={id(p) for p in bp}; hp=[p for p in model.parameters() if id(p) not in ids]
        opt=torch.optim.AdamW([{"params":bp,"lr":CFG["lr"]*CFG["backbone_lr_scale"]},{"params":hp,"lr":CFG["lr"]}],weight_decay=CFG["weight_decay"])
        scaler=torch.amp.GradScaler("cuda",enabled=CFG["mixed_precision"] and DEVICE.type=="cuda")
        va=None
        for ep in range(1,CFG["tune_epochs"]+1):
            run_epoch(model,train_loader,opt,scaler,ep);va=run_epoch(model,val_loader,None,None,ep)
            trial.report(float(np.nan_to_num(va["auroc"],nan=0)),ep)
            if trial.should_prune(): raise optuna.TrialPruned()
        score=.50*np.nan_to_num(va["auroc"],nan=0)+.25*np.nan_to_num(va["auprc"],nan=0)+.10*va["dice_disc"]+.10*va["dice_cup"]-.05*va["ece"]
        del model;gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return float(score)
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=CFG["seed"]),pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study.optimize(objective,n_trials=CFG["optuna_trials"],show_progress_bar=False)
    trials=study.trials_dataframe();STORE.save_df(trials,f"folds/{target_source}/optuna_trials.csv")
    best=study.best_params;STORE.save_json(best,f"folds/{target_source}/best_hyperparameters.json")
    for k,v in base.items():CFG[k]=v
    return best

# ---------------------------- 15. MAIN FOLD LOOP ----------------------------
def train_fold(target_source):
    fold_rel=f"folds/{target_source}";local_fold,drive_fold=STORE.dirs(fold_rel)
    completed=find_existing(f"{fold_rel}/COMPLETED.json")
    if CFG["resume"] and completed:
        saved=json.loads(Path(completed).read_text());pred=pd.read_csv(find_existing(f"{fold_rel}/test_predictions.csv"))
        return saved["metrics"],pred,None

    train_pool=META[META.source!=target_source].copy();test_df=META[META.source==target_source].copy()
    train_df,val_df=source_stratified_split(train_pool,CFG["val_fraction"],CFG["seed"])
    if CFG["fast_dev_run"]:
        train_df=train_df.groupby(["source","label"],group_keys=False).head(8).reset_index(drop=True)
        val_df=val_df.groupby(["source","label"],group_keys=False).head(4).reset_index(drop=True)
        test_df=test_df.groupby(["source","label"],group_keys=False).head(6).reset_index(drop=True)
    STORE.save_df(train_df,f"{fold_rel}/train_split.csv");STORE.save_df(val_df,f"{fold_rel}/val_split.csv");STORE.save_df(test_df,f"{fold_rel}/test_split.csv")
    base_fold_cfg={k:CFG[k] for k in ["lr","weight_decay","dropout","lambda_proto","lambda_cons","lambda_domain"]}
    tuned=tune_fold_hyperparameters(train_df,val_df,target_source)
    for k,v in tuned.items(): CFG[k]=v
    STORE.save_json({k:CFG[k] for k in base_fold_cfg},f"{fold_rel}/effective_hyperparameters.json")
    train_loader=make_loader(train_df,True);val_loader=make_loader(val_df,False);test_loader=make_loader(test_df,False)
    model=RimGraphDG(num_domains=train_df.source.nunique()).to(DEVICE)
    backbone_params=list(model.backbone.parameters());backbone_ids={id(p) for p in backbone_params};head_params=[p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer=torch.optim.AdamW([
        {"params":backbone_params,"lr":CFG["lr"]*CFG["backbone_lr_scale"]},
        {"params":head_params,"lr":CFG["lr"]}],weight_decay=CFG["weight_decay"])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(1,CFG["epochs"]))
    scaler=torch.amp.GradScaler("cuda",enabled=CFG["mixed_precision"] and DEVICE.type=="cuda")
    history=[];best_score=-np.inf;best_epoch=0;start_epoch=1
    last=find_existing(f"{fold_rel}/last_checkpoint.pt")
    if CFG["resume"] and last:
        ck=torch.load(last,map_location=DEVICE);model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"]);scheduler.load_state_dict(ck["scheduler"]);history=ck["history"];best_score=ck["best_score"];best_epoch=ck["best_epoch"];start_epoch=ck["epoch"]+1
    for epoch in range(start_epoch,CFG["epochs"]+1):
        tr=run_epoch(model,train_loader,optimizer,scaler,epoch);va=run_epoch(model,val_loader,None,None,epoch)
        scheduler.step();score=np.nan_to_num(va["auroc"],nan=0)*.55+np.nan_to_num(va["auprc"],nan=0)*.25+va["dice_disc"]*.10+va["dice_cup"]*.10
        row={"epoch":epoch,"train_loss":tr["loss"],"val_loss":va["loss"],"val_auroc":va["auroc"],"val_auprc":va["auprc"],"val_f1":va["f1"],"val_dice_disc":va["dice_disc"],"val_dice_cup":va["dice_cup"],"selection_score":score,"lr":optimizer.param_groups[-1]["lr"]}
        history.append(row);STORE.save_df(pd.DataFrame(history),f"{fold_rel}/history.csv")
        ck={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"history":history,"best_score":best_score,"best_epoch":best_epoch,"cfg":CFG}
        if CFG["save_every_epoch"]:save_checkpoint(ck,f"{fold_rel}/last_checkpoint.pt")
        if score>best_score:
            best_score=score;best_epoch=epoch;save_checkpoint({**ck,"best_score":best_score,"best_epoch":best_epoch},f"{fold_rel}/best_model.pt")
        DASH.update(target_source,epoch,history,note=f"Best source-validation score: **{best_score:.4f}** at epoch **{best_epoch}**")
        if epoch-best_epoch>=CFG["patience"]:break
    best=torch.load(find_existing(f"{fold_rel}/best_model.pt"),map_location=DEVICE);model.load_state_dict(best["model"])
    val_pred,_=predict(model,val_loader)
    T_standard=fit_temperature(val_pred.label.values,val_pred.prob_raw.values)
    T_robust=fit_robust_temperature(val_pred.label.values,val_pred.prob_raw.values,val_pred.source.values)
    cal_rows=[]
    for name,Tc in [("standard",T_standard),("robust",T_robust)]:
        pc=apply_temperature(val_pred.prob_raw.values,Tc);mm=classification_metrics(val_pred.label.values,pc,.5)
        worst_nll=max(classification_metrics(g.label.values,apply_temperature(g.prob_raw.values,Tc),.5)["nll"] for _,g in val_pred.groupby("source"))
        cal_rows.append({"method":name,"temperature":Tc,"pooled_nll":mm["nll"],"pooled_ece":mm["ece"],"worst_source_nll":worst_nll})
    STORE.save_df(pd.DataFrame(cal_rows),f"{fold_rel}/calibration_selection.csv")
    T=T_robust
    val_cal=apply_temperature(val_pred.prob_raw.values,T);thr=optimal_threshold(val_pred.label.values,val_cal)
    test_pred,segm=predict(model,test_loader);test_pred["prob_calibrated"]=apply_temperature(test_pred.prob_raw.values,T)
    metrics=classification_metrics(test_pred.label.values,test_pred.prob_calibrated.values,thr);metrics.update(segm);metrics.update({"held_out_source":target_source,"temperature":T,"temperature_standard":T_standard,"calibration_method":"worst-source temperature scaling","best_epoch":best_epoch,"n_test":len(test_pred)})
    STORE.save_df(test_pred,f"{fold_rel}/test_predictions.csv");STORE.save_json(metrics,f"{fold_rel}/metrics.json")
    save_evaluation_figures(test_pred,metrics,target_source);save_visual_examples(model,test_df,target_source,CFG["n_visual_examples"])
    sector_cols=[f"sector_{j+1:02d}" for j in range(CFG["num_sectors"])]
    sector_summary=test_pred.groupby("label")[sector_cols].mean().T.reset_index().rename(columns={"index":"sector",0:"normal_mean",1:"glaucoma_mean"})
    STORE.save_df(sector_summary,f"{fold_rel}/xai_sector_summary.csv")
    STORE.save_json({"metrics":metrics,"completed_at":time.strftime("%Y-%m-%d %H:%M:%S")},f"{fold_rel}/COMPLETED.json")
    for k,v in base_fold_cfg.items(): CFG[k]=v
    return metrics,test_pred,model

all_metrics=[];all_predictions=[]
for target in CFG["fold_targets"]:
    m,p,_=train_fold(target);all_metrics.append(m);all_predictions.append(p);DASH.fold_rows=[fmt_metrics(x)|{"held_out_source":x["held_out_source"]} for x in all_metrics]
    DASH.update(target,m.get("best_epoch",0),[],note="Fold complete.")
    gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ---------------------------- 16. FINAL REPORT ----------------------------
summary=pd.DataFrame(all_metrics)
preds=pd.concat(all_predictions,ignore_index=True)
STORE.save_df(summary,"fold_summary.csv");STORE.save_df(preds,"all_external_predictions.csv")
metric_cols=["auroc","auprc","accuracy","balanced_accuracy","sensitivity","specificity","f1","mcc","brier","ece","dice_disc","dice_cup"]
aggregate=[]
for c in metric_cols:
    aggregate.append({"metric":c,"mean":summary[c].mean(),"std":summary[c].std(ddof=1),"worst_domain":summary[c].min() if c not in ["brier","ece"] else summary[c].max()})
agg=pd.DataFrame(aggregate);STORE.save_df(agg,"aggregate_metrics.csv")

# Manifest and checksums
manifest=[]
for p in sorted(LOCAL_RUN.rglob("*")):
    if p.is_file():
        h=hashlib.sha256(p.read_bytes()).hexdigest();manifest.append({"file":str(p.relative_to(LOCAL_RUN)),"bytes":p.stat().st_size,"sha256":h})
STORE.save_df(pd.DataFrame(manifest),"artifact_manifest.csv")

zip_path=shutil.make_archive(str(LOCAL_RUN),"zip",root_dir=LOCAL_RUN)
shutil.copy2(zip_path,DRIVE_RUN.parent/f"{RUN_ID}.zip")

clear_output(wait=True)
display(Markdown("# ✅ RimGraph-DG run completed"))
display(Markdown(f"**Colab artifacts:** `{LOCAL_RUN}`  \n**Drive artifacts:** `{DRIVE_RUN}`  \n**Bundle:** `{DRIVE_RUN.parent / (RUN_ID + '.zip')}`"))
display(Markdown("## External fold results"));display(summary[["held_out_source"]+metric_cols+["temperature","best_epoch","n_test"]].style.format(precision=4).hide(axis="index"))

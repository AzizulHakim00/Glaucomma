display(Markdown("## Aggregate and worst-domain results"));display(agg.style.format(precision=4).hide(axis="index"))

fig=plt.figure(figsize=(14,5));
x=np.arange(len(summary));w=.22
for i,c in enumerate(["auroc","auprc","balanced_accuracy","f1"]):plt.bar(x+(i-1.5)*w,summary[c],width=w,label=c)
plt.xticks(x,summary.held_out_source);plt.ylim(0,1.02);plt.ylabel("Score");plt.title("Leave-one-dataset-out external performance");plt.legend();plt.grid(axis="y",alpha=.2);plt.tight_layout();STORE.save_figure(fig,"external_fold_comparison.png");display(fig);plt.close(fig)

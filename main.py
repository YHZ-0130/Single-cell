import urllib.request
import anndata
import pandas as pd
from pathlib import Path
from ikarusCNN import classifier, utils, data

Path("data").mkdir(exist_ok=True)

if not Path("data/laughney20_lung/adata.h5ad").is_file():
    Path("data/laughney20_lung").mkdir(exist_ok=True)
    urllib.request.urlretrieve(
        url="https://bimsbstatic.mdc-berlin.de/akalin/Ikarus/part_1/data/laughney20_lung/adata.h5ad",
        filename=Path("data/laughney20_lung/adata.h5ad")
    )

if not Path("data/lee20_crc/adata.h5ad").is_file():
    Path("data/lee20_crc").mkdir(exist_ok=True)
    urllib.request.urlretrieve(
        url="https://bimsbstatic.mdc-berlin.de/akalin/Ikarus/part_1/data/lee20_crc/adata.h5ad",
        filename=Path("data/lee20_crc/adata.h5ad")
    )

if not Path("data/tirosh17_headneck/adata.h5ad").is_file():
    Path("data/tirosh17_headneck").mkdir(exist_ok=True)
    urllib.request.urlretrieve(
        url="https://bimsbstatic.mdc-berlin.de/akalin/Ikarus/part_1/data/tirosh17_headneck/adata.h5ad",
        filename=Path("data/tirosh17_headneck/adata.h5ad")
    )

paths = [
    Path("data/laughney20_lung/"),
    Path("data/lee20_crc/"),
    Path("data/tirosh17_headneck/")
]
names = [
    "laughney",
    "lee",
    "tirosh"
]
adatas = {}
for path, name in zip(paths, names):
    print(path / "adata.h5ad")
    adatas[name] = anndata.read_h5ad(path / "adata.h5ad")
    # Uncomment to perform preprocessing. Here, the loaded anndata objects are already preprocessed.
    # adatas[name] = data.preprocess_adata(adatas[name])

print(adatas["lee"].obs.head(2))

print(adatas["lee"].var.head(2))

signatures_path = Path("out/signatures.gmt")
pd.read_csv(signatures_path, sep="\t", header=None)
model = classifier.Ikarus(signatures_gmt=str(signatures_path), out_dir="out")

train_adata_list = [adatas["laughney"], adatas["lee"]]
train_names_list = ["laughney", "lee"]
obs_columns_list = ["tier_0_hallmark_corrected", "tier_0_hallmark_corrected"]

model.fit(train_adata_list, train_names_list, obs_columns_list, save=True)
_ = model.predict(adatas["tirosh"], "tirosh", save=True)
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_confusion_matrix(
        y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, ax=None
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.rcParams["figure.figsize"] = [6, 4]
    # print(classes)
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred, labels=classes)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    if ax is None:
        (fig, ax) = plt.subplots()

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
    ):
        item.set_fontsize(12)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    return fig, ax


_ = model.get_umap(adatas["tirosh"], "tirosh", save=True)
path = Path("out/tirosh")
results = pd.read_csv(path / "prediction.csv", index_col=0)
adata = anndata.read_h5ad(path / "adata_umap.h5ad")
y = adata.obs.loc[:, "tier_0"]
y_pred_lr = results["final_pred"]
acc = metrics.balanced_accuracy_score(y, y_pred_lr)
print(metrics.classification_report(y, y_pred_lr, labels=["Normal", "Tumor"]))
fig, ax = plot_confusion_matrix(
    y,
    y_pred_lr,
    classes=["Normal", "Tumor"],
    title=f"confusion matrix \n train on laughney+lee, test on tirosh \n balanced acc: {acc:.3f}",
)
fig.tight_layout()

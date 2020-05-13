# %%
from sklearn.metrics import classification_report, confusion_matrix
import torch
from detector.sgcn import SignedGraphConvolutionalNetwork
from core.config import cfg


# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SignedGraphConvolutionalNetwork(
    device, X.shape).to(device)

epoch_to_load=100
checkpoint_filename = "checkpoint_epoch_{}.pth".format(epoch_to_load)
checkpoint_filepath = os.path.join(cfg.DATA.CHECKPOINT_DIR, checkpoint_filename)
model.load_state_dict(torch.load(checkpoint_filepath))

# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# %%
model.eval()
y_label = []
y_predict = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        images, labels = data
        N = images.size(0)
        images = Variable(images).to(device)
        outputs = model(images)
        prediction = outputs.max(1, keepdim=True)[1]
        y_label.extend(labels.cpu().numpy())
        y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

# compute the confusion matrix
confusion_mtx = confusion_matrix(y_label, y_predict)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, cfg.CLASSES)
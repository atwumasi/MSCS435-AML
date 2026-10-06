import matplotlib.pyplot as plt
import torch


def plot_loss_and_accuracy(train_loss_list, valid_loss_list, train_acc_list, valid_acc_list):
    """Side-by-side loss and accuracy curves, for a quick bias/variance check."""
    fig, axis = plt.subplots(1, 2, figsize=(20, 5))
    axis[0].plot(train_loss_list, color='r', label='train loss')
    axis[0].plot(valid_loss_list, color='b', label='dev loss')
    axis[0].set_title('Loss Comparison')
    axis[0].legend()
    axis[1].plot(train_acc_list, color='r', label='train accuracy')
    axis[1].plot(valid_acc_list, color='b', label='dev accuracy')
    axis[1].set_title('Accuracy Comparison')
    axis[1].legend()
    return fig, axis


def visualize_results(model, X_valid_t, X_valid, y_valid, device, index):
    """Compare the actual vs. predicted segmentation mask for a single validation example."""
    model.eval()
    img = X_valid_t[index:index + 1].to(device)  # keep batch dim: (1, C, H, W)
    with torch.no_grad():
        pred_logits = model(img)
    pred_mask = torch.argmax(pred_logits[0], dim=0).cpu().numpy()  # (H, W)

    fig, arr = plt.subplots(1, 3, figsize=(15, 15))
    arr[0].imshow(X_valid[index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y_valid[index, :, :, 0])
    arr[1].set_title('Actual Masked Image')
    arr[2].imshow(pred_mask)
    arr[2].set_title('Predicted Masked Image')
    return fig, arr

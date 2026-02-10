
import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


# Plot the function (curve + optional scatter)
def plot_fun(x=None, y=None, x_scatter=None, y_scatter=None):
    fig = go.Figure()

    # Plot continuous curve
    if x is not None and y is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="f(x) (true function)",
                line=dict(width=3, color='blue')
            )
        )

    # Plot scattered points
    if x_scatter is not None and y_scatter is not None:
        fig.add_trace(
            go.Scatter(
                x=x_scatter,
                y=y_scatter,
                mode="markers",
                name="f(x) (samples)",
                marker=dict(size=8, color='red')
            )
        )

    fig.update_layout(
        title="Non-linear Function Visualization",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        width=800,
        height=450
    )

    fig.show()
  

def train_MLP(model, n_epochs, train_dataloader, loss_fn):
    # Define the loss function and optimizer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    nTrainSteps = n_epochs

    # Run the training loop
    for epoch in range(0, nTrainSteps):

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(train_dataloader, 0):
            # Get inputs
            inputs, targets = data
            
            # Zero the gradients
            optimizer.zero_grad()
            # Perform forward pass (make sure to supply the input in the right way)
            outputs = model(inputs)
            # Compute loss
            loss = loss_fn(outputs, targets)
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
            # Print statistics
            current_loss += loss.item()

        if (epoch + 1) % 1000 == 0:
            print('Loss after epoch %5d: %.3f' %
                    (epoch + 1, current_loss))
            current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc



# training function (see later)
def train_model(model, epochs, X_train, X_test, y_train, y_test, loss_fn):
    print("train:")
    
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    
    for epoch in range(epochs):
        model.train()

        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_train, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_logits = model(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_test, test_pred)

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Loss {loss:.4f} Acc {acc:.2f}% | Test {test_loss:.4f} {test_acc:.2f}%")
   
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

import torch

import matplotlib.pyplot as plt

from nn_fct import FullyConnectedNet, convert_dataset, train_test, validate_model, load_dataset

def visualize_results(train_results, test_results):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot 1
    axs[0, 0].plot(train_results[0])
    axs[0, 0].set_title('Training MSE')

    # Plot 2
    axs[0, 1].plot(train_results[1])
    axs[0, 1].set_title('Training MAE')

    # Plot 3
    axs[1, 0].plot(test_results[0])
    axs[1, 0].set_title('Testing MSE')

    # Plot 4
    axs[1, 1].plot(test_results[1])
    axs[1, 1].set_title('Testing MAE')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x_train, y_train, x_test, y_test, x_val, y_val = load_dataset("data/final_datav2.csv",0.3,True)

    batch_size = 32
    lr = 0.001

    train_loader = convert_dataset(x_train, y_train, batch_size, device)
    test_loader = convert_dataset(x_test, y_test, batch_size, device)
    val_loader = convert_dataset(x_val, y_val, batch_size, device)

    model = FullyConnectedNet(
        input_dim=x_train.shape[1],
        output_dim=1,
        width=32,
        depth=4,
    ).to(device)

    (model, state_model), train_results, test_results = train_test(
        model=model,
        learning_rate=lr,
        train_loader=train_loader,
        test_loader=test_loader,
        patience=10,
        num_epochs=200,
        device=device
    )

    torch.save(state_model, "best_model.pth")

    val_mse, val_mae = validate_model(model,val_loader, device)
    print(f"Final MSE : {val_mse}")
    print(f"Final MAE : {val_mae}")

    visualize_results(train_results, test_results)




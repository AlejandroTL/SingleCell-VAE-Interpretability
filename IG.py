import argparse
from general import *
from captum.attr import IntegratedGradients

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # VAE Definition
    parser.add_argument("--model_path", help="model path")

    # Data
    parser.add_argument("--data_path", help="path to data csv")

    args = parser.parse_args()

    PATH = args.model_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=2000, mid_dim=512, features=32, drop=0.3)
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    model.eval()

    data_path = args.data_path

    df = pd.read_csv(data_path, header=0)
    df_t = df.T
    df_t.columns = df['Unnamed: 0']
    df_t = df_t.drop(['Unnamed: 0'])
    df_t.reset_index(drop=True, inplace=True)
    df_t_float = df_t.astype(float, copy=True)

    tensor_data = torch.tensor(df_t_float.values).to(device)
    reconstruction, _, _, latent_space = model(tensor_data.float())

    input_es = latent_space[0:7000].detach().requires_grad_().to(device)
    gradients_summation = []
    baseline = latent_space.mean(axis=0).to(device)

    for i in range(model.decoder[3].out_features):
        print(i)
        ig = IntegratedGradients(model.decoder, True)
        attributions = ig.attribute(input_es.to(device), baselines=baseline.view(1, 32).to(device), target=i)
        gradients_summation.append(torch.sum(attributions, axis=0).detach().cpu().numpy())

    sum_df = pd.DataFrame(gradients_summation).set_index(df_t_float.columns)

    sum_df.to_csv("IG.csv")

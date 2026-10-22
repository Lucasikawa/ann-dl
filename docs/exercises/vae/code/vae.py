"""VAE em PyTorch (Exercícios 1 e 2).

O autograd calcula os gradientes; as partes específicas do VAE são escritas à mão:
  * a reparametrização  z = mu + sigma * eps,  eps ~ N(0, I);
  * o termo de reconstrução (BCE somada nos 784 pixels, média nas imagens);
  * o termo KL em forma fechada, KL(N(mu, sigma^2) || N(0, I));
  * a amostragem da priori, z ~ N(0, I), decodificada.
Nada vem de uma classe de VAE pronta nem de torch.distributions.
"""
import torch
import torch.nn.functional as F
from torch import nn

N_PIXELS = 784


class VAE(nn.Module):
    """Arquitetura do enunciado: 784 -> 512 -> 256 -> (mu, log sigma^2) em R^d -> 256 -> 512 -> 784."""

    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(N_PIXELS, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, latent_dim)       # média de q(z|x)
        self.logvar_head = nn.Linear(256, latent_dim)   # log sigma^2 de q(z|x): qualquer real é válido
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, N_PIXELS), nn.Sigmoid(),     # cada pixel = probabilidade de estar "aceso"
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, generator=None):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar, generator)
        return self.decode(z), mu, logvar


def reparameterize(mu, logvar, generator=None):
    """z = mu + sigma * eps. O sorteio fica em eps, que não depende dos parâmetros,
    então o gradiente passa por mu e por sigma = exp(logvar / 2)."""
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
    return mu + sigma * eps


def reconstruction_term(x_hat, x):
    """-log p(x|z) com pixels Bernoulli: BCE SOMADA nos 784 pixels, dividida pelo nº de imagens."""
    return F.binary_cross_entropy(x_hat, x, reduction="sum") / x.shape[0]


def kl_term(mu, logvar):
    """KL(N(mu, diag sigma^2) || N(0, I)) em forma fechada, somada nas d dimensões,
    média nas imagens:  1/2 * sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2)."""
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / mu.shape[0]


def kl_per_dimension(mu, logvar):
    """O mesmo KL, mas sem somar nas dimensões (média nas imagens): quanto cada dimensão "custa"."""
    return 0.5 * torch.mean(mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=0)


@torch.no_grad()
def sample_prior(model, n):
    """n imagens decodificadas de z ~ N(0, I). Usa o gerador global: chame torch.manual_seed antes."""
    z = torch.randn(n, model.latent_dim)
    return model.decode(z)


def run_epoch(model, X, batch_size, optimizer=None, beta=1.0, generator=None):
    """Uma passada por X. Com optimizer, treina (lotes embaralhados); sem, só avalia.

    Devolve (reconstrução, KL) em nats por imagem: médias ponderadas pelo tamanho de cada lote,
    ou seja, a média sobre todas as imagens da passada.
    """
    training = optimizer is not None
    model.train(training)
    n = X.shape[0]
    order = torch.randperm(n) if training else torch.arange(n)
    rec_total = kl_total = 0.0
    with torch.set_grad_enabled(training):
        for start in range(0, n, batch_size):
            x = X[order[start:start + batch_size]]
            x_hat, mu, logvar = model(x, generator)
            rec = reconstruction_term(x_hat, x)
            kl = kl_term(mu, logvar)
            if training:
                loss = rec + beta * kl          # -ELBO do lote, por imagem
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            rec_total += rec.item() * x.shape[0]
            kl_total += kl.item() * x.shape[0]
    return rec_total / n, kl_total / n


def evaluate(model, X, seed=0, batch_size=1000):
    """Reconstrução e KL num conjunto, com UM z sorteado por imagem (estimativa usual do ELBO).

    O ruído vem de um gerador próprio com seed fixa: a avaliação não mexe no gerador global
    (que embaralha os lotes do treino), e todas as avaliações usam o mesmo ruído.
    """
    return run_epoch(model, X, batch_size, generator=torch.Generator().manual_seed(seed))


def train(model, X_train, X_val, epochs=20, lr=1e-3, batch_size=128, beta=1.0):
    """Adam, lr 1e-3, lotes de 128. Registra os dois termos no treino (média da época) e na validação."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    hist = {"train_rec": [], "train_kl": [], "val_rec": [], "val_kl": []}
    hist["val_rec_epoch0"], hist["val_kl_epoch0"] = evaluate(model, X_val)  # antes de treinar
    for _ in range(epochs):
        rec, kl = run_epoch(model, X_train, batch_size, optimizer, beta)
        hist["train_rec"].append(rec)
        hist["train_kl"].append(kl)
        rec, kl = evaluate(model, X_val)
        hist["val_rec"].append(rec)
        hist["val_kl"].append(kl)
    return hist

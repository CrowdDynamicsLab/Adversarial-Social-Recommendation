from VAE import MultiVAE
import os


def generator_vaecf(n_items, n_users, rating_matrix, p_dims, q_dims):
    total_anneal_steps = 20000  # VAECF recommended values
    anneal_cap = 0.2  # VAECF recommended values

    # tf.reset_default_graph()
    vae = MultiVAE(p_dims, n_users, rating_matrix, q_dims=q_dims, lam=0.0, random_seed=98765)

    logits_var, loss_var, params, user_emb = vae.build_graph()

    return vae, logits_var, loss_var, params, total_anneal_steps, anneal_cap, user_emb

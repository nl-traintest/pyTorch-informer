from sklearn.manifold import TSNE

# ----- dimension reduction by T-SNE
def TSNE_to_2D(input_embeddings):
    # [batch, high_dimensions] -> [batch, 2]
    tsne = TSNE(n_components=2, random_state=42, perplexity=2)
    embedding_2d = tsne.fit_transform(input_embeddings)
    return embedding_2d

import numpy as np

class Search:
    # Returns top 5 closest vectors
    @staticmethod
    def query(query, features, img_paths):
        dists = np.linalg.norm(features - query, axis=1)
        ids = np.argsort(dists)[:5]
        images = [img_paths[id] for id in ids]

        return images
import scipy.sparse as sp
import numpy as np

def give_recommendations(
    model,
    user,
    session_history,
    user_to_idx,
    track_to_idx,
    idx_to_track,
    user_track_matrix,
    K=10
):
    cols = [track_to_idx[t] for t in session_history if t in track_to_idx]

    session_prefix_vec = sp.csr_matrix(
        (np.ones(len(cols)), ([0] * len(cols), cols)),
        shape=(1, len(track_to_idx))
    )

    if user not in user_to_idx:
        recs, _ = model.recommend(
            userid=0,
            user_items=session_prefix_vec,
            N=K,
            recalculate_user=True,
            filter_already_liked_items=True
        )
    else:
        user_id = user_to_idx[user]
        user_vec = user_track_matrix[user_id] + session_prefix_vec

        recs, _ = model.recommend(
            userid=user_id,
            user_items=user_vec,
            N=K,
            recalculate_user=True,
            filter_already_liked_items=True
        )

    return [idx_to_track[int(i)] for i in recs]


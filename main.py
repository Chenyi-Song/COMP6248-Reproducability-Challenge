import train_pep as pep
import train_bl as bl

from data import get_dataset
from model import get_model, trans_pep


if __name__ == '__main__':
    # train_movielens()
    
    dataset_name = 'movielens'
    dataset_path = './data/ml-1m/'
    dataset = get_dataset(dataset_name, dataset_path)

    ml_emb_dim = 32

    # FM
    bl_fm = get_model('fm', dataset, ml_emb_dim)
    print('FM', bl_fm)
    
    # DeepFM
    bl_dfm = get_model('dfm', dataset, ml_emb_dim,
                       dfm_mlp_dims=(100, 100))
    print('DeepFM', bl_dfm)

    # AutoInt
    bl_afi = get_model('afi', dataset, ml_emb_dim,
                       afi_mlp_dims=(100, 100), afi_drop_out=(0.4, 0.4, 0.4))
    print('AutoInt', bl_afi)

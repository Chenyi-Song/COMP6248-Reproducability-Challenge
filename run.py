from data import get_dataset
from model import get_model, trans_pep

import train_bl as bl
import train_pep as pep

def train_fm():
    dataset_name = 'movielens'
    dataset_path = './data/ml-1m/'
    dataset = get_dataset(dataset_name, dataset_path)

    ml_emb_dim = 32
    epoch = 15

    print('*' * 10, 'Train PEP FM on MovieLens', '*' * 10)
    # FM
    pep_fm = get_model('fm', dataset, ml_emb_dim)
    trans_pep(pep_fm, dataset.field_dims, ml_emb_dim)
    emb_nums = pep.train(dataset, pep_fm,  path='./fm-log/pep-movielens/{file_name}')

    for num in emb_nums:
        print('*' * 10, f' Retrain FM on MovieLens (Param: {num}) ', '*' * 10)
        retrain_fm = get_model('fm', dataset, ml_emb_dim)
        trans_pep(retrain_fm, dataset.field_dims, ml_emb_dim,
                  retrain=True, emb_save_path='fm-log/pep-movielens/embedding-{num}.npy', retrain_emb_param=num)
        pep.retrain(dataset, retrain_fm, num, epoch_size=epoch,
                    path='./fm-log/retrain-movielens/{file_name}')
    
    print('*' * 10, 'Train Baseline FM on MovieLens', '*' * 10)
    # FM
    bl_fm = get_model('fm', dataset, ml_emb_dim)
    bl.train(dataset, bl_fm, epoch_size=epoch, log_path='./fm-log/bl-movielens/{file_name}')


def train_dfm():
    dataset_name = 'movielens'
    dataset_path = './data/ml-1m/'
    dataset = get_dataset(dataset_name, dataset_path)

    ml_emb_dim = 32
    epoch = 15

    print('*' * 10, 'Train PEP DeepFM on MovieLens', '*' * 10)
    # SeepFM
    pep_dfm = get_model('dfm', dataset, ml_emb_dim,
                        dfm_mlp_dims=(100, 100))
    trans_pep(pep_dfm, dataset.field_dims, ml_emb_dim)
    emb_nums = pep.train(dataset, pep_dfm, path='./dfm-log/pep-movielens/{file_name}')

    # DeepFM
    for num in emb_nums:
        print('*' * 10, f' Retrain DeepFM on MovieLens (Param: {num}) ', '*' * 10)
        retrain_dfm = get_model('dfm', dataset, ml_emb_dim,
                                dfm_mlp_dims=(100, 100))
        trans_pep(retrain_dfm, dataset.field_dims, ml_emb_dim,
                  retrain=True, emb_save_path='dfm-log/pep-movielens/embedding-{num}.npy', retrain_emb_param=num)
        pep.retrain(dataset, retrain_dfm, num, epoch_size=epoch,
                    path='./dfm-log/retrain-movielens/{file_name}')

    print('*' * 10, 'Train Baseline DeepFM on MovieLens', '*' * 10)
    # DeepFM
    bl_dfm = get_model('dfm', dataset, ml_emb_dim, dfm_mlp_dims=(100, 100))
    bl.train(dataset, bl_dfm, epoch_size=epoch, log_path='./dfm-log/bl-movielens/{file_name}')



def train_ati(): 
    dataset_name = 'movielens'
    dataset_path = './data/ml-1m/'
    dataset = get_dataset(dataset_name, dataset_path)

    ml_emb_dim = 32
    epoch = 15

    print('*' * 10, 'Train PEP AutoInt on MovieLens', '*' * 10)
    # AutoInt
    pep_afi = get_model('afi', dataset, ml_emb_dim,
                        afi_mlp_dims=(100, 100), afi_drop_out=(0.4, 0.4, 0.4))
    trans_pep(pep_afi, dataset.field_dims, ml_emb_dim)
    emb_nums = pep.train(dataset, pep_afi, path='./afi-log/pep-movielens/{file_name}')

    # AutoInt
    for num in emb_nums:
        print('*' * 10, f' Retrain AutoInt on MovieLens (Param: {num}) ', '*' * 10)
        retrain_afi = get_model('afi', dataset, ml_emb_dim,
                                afi_mlp_dims=(100, 100), afi_drop_out=(0.4, 0.4, 0.4))
        trans_pep(retrain_afi, dataset.field_dims, ml_emb_dim,
                  retrain=True, emb_save_path='afi-log/pep-movielens/embedding-{num}.npy', retrain_emb_param=num)
        pep.retrain(dataset, retrain_afi, num, epoch_size=epoch,
                    path='./afi-log/retrain-movielens/{file_name}')

    print('*' * 10, 'Train Baseline AutoInt on MovieLens', '*' * 10)
    # AutoInt
    bl_afi = get_model('afi', dataset, ml_emb_dim,
                       afi_mlp_dims=(100, 100), afi_drop_out=(0.4, 0.4, 0.4))
    bl.train(dataset, bl_afi, epoch_size=epoch,
             log_path='./afi-log/bl-movielens/{file_name}')


if __name__ == '__main__':
    train_fm()

    train_dfm()

    # train_ati()
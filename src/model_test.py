from oncoGAN import simulate_counts
import glob

def test_models(models: list[str], save_csv = 1, n = 1, count = 110):
    for model in models:
        for _ in range(n):
            simulate_counts('Breast-AdenoCa', count, model, save_csv=save_csv)
        print(f'finished testing {model}', flush=True)


# models = [x.split('ics_')[1] for x in glob.glob('/home/run/runs/donor_*')]
# models = ['default_k-10_so', 'default_k-1000_mo', 'default_k-1000_so', 'brca-path_k-1000_mo', 'brca-path_k-10_so', 'brca-path_k-1000_so', 'brca-path_k-100_so', 'default_k-100_so']
models = [None]

test_models(models, save_csv=1, n=100, count = 197)
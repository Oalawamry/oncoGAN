from oncoGAN import simulate_counts
import glob

def test_models(models: list[str]):
    for model in models:
        simulate_counts('Breast-AdenoCa', 100, model)
        print(f'finished testing {model}')


models = [x.split('ics_')[1] for x in glob.glob('/home/run/runs/donor_*')]

test_models(models)
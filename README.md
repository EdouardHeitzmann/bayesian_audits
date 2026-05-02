# bayesian_audits

This is minimal replication code for my Bayesian Stats final project. This repo is still quite raw and poorly organized -- my apologies -- but most of the content in my writeup can be replicated using the notebook `adversarial_profile` ipynb. I will definitely be revisiting and refining this repository in the future.

Also included:
- `scottish_profile.ipynb` (uses local data path `data/scot-elex/6_cands/aberdeenshire_2012_ward9.csv`). This notebook might not run on a fresh install of votekit, as I had to tweak some of the internals of my FastSTV code to optimize the 6 candidate audits.
- `src/` (core Python code used by the notebooks)
- `requirements.txt` 

To run locally:
1. Create/activate a Python environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `adversarial_profile.ipynb`.

If Prof Dukic is readin this, I thank you for this semester! This has been my first formal introduction to Bayesian Stats, and it has been very productive for me.
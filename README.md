# Non-Trivial-Repository

Developing a Multimodal AI bot for Seizure Forecasting

Current bot files are available on repository as .joblib files.

To train, run appropriate bot file

For ECG:

Run botECGBinary.py

For EMG:

Run botEMGBinary.py

For EEG:

Run botEEG.py for generalized training.
Then run FINAL_EEG_BOT.py for personalized training (currently personalizing on PN00)

Note: remove PN00 files from Siena dataset before personalizing, and then add back when running second file

BotEMG and BotECG.py are old files which would also attempt to guess time period instead of just detecting abnormalities.

Datasets are not included for memory purposes. Datasets used are: Siena-Scalp-EEG Dataset, and SeizeIT2 Dataset.
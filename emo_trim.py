import pandas as pd

def select_emotion(emotions, timeframe, pref_emo):
    df = pd.DataFrame({"timeframe": timeframe, "emotions": emotions})

    trimmed = df[df["emotions"]==pref_emo]

    return trimmed['timeframe'].tolist()
import utils

if __name__ == "__main__":
    datapath = "./data/val_in/val_in"
    outpath = "./output/"
    MIAname = "LSTM_MIA.csv"
    PITname = "LSTM_PIT.csv"
    mergeName = "LTSM.csv"
    merge_output = utils.merge_output(datapath, outpath, MIAname, PITname, mergeName)

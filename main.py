# Authors: Andrew Wernersbach, Jacob Zaslav



import API_client
import data_processing
import train

def main():
    df_test = data_processing.read()
    #df_test = df_test.sample(n=100000)
    df_test = df_test.sample(frac=5/8, random_state=1).reset_index()
    [X_train, X_val, y_train, y_val] = data_processing.preprocess(df_test)
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    train.train(X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    main()
    


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def load_dataset(path: str = 'Path_to_csv'):
    '''
    This function is used to load csv dataset file into a dataframe
    Parameters
    ----------
    path : str, optional
        Path to the CSV dataset File . The default is 'Path_to_csv'.
    Returns
    -------
    df : DataFrame with features as columns.
    '''
    
    df = pd.read_csv(path)
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df
def split_predictors_target(data: pd.DataFrame = None, target: str = 'estimated_stock_pct'):
    '''
    This Function splits the datasets into two dataset X,y one contains the predictors (X)
    and the other contains the target variable (y)
    Parameters
    ----------
    data : pd.DataFrame, optional
        Dataframe with predictors and target variable. The default is None.
    target : str, optional
        Target column's name. The default is 'estimated_stock_pct'.

    Raises
    ------
    Exception
        The user entered a non valid target column's name.

    Returns
    -------
    X : pd.DataFrame
        Dataframe of predictor features.
    y : pd.DataFrame
        Dataframe of Target vaiable.
    '''
    if target not in data.columns:
        raise Exception(f'Target{target} is not present in Dataframe columns, Please enter valid target name.')
    X = data.drop(columns=[target])
    y = data[target]
    return X,y

def train_with_split(X: pd.DataFrame = None,y: pd.DataFrame = None,test_split: float = 0.2):
    '''

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe of predictor features.
    y : pd.DataFrame
        Dataframe of Target vaiable.
    test_split : float
        fraction of data to holdout as test data.
    Returns
    -------
    mean absolute error of train data.
    mean absolute error of test data.
    '''
    RFR = RandomForestRegressor(max_depth=5, min_samples_split=25, n_estimators=25,
                      n_jobs=-1, random_state=42)
    SS = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
    X_train = SS.fit_transform(X_train)
    X_test = SS.fit_transform(X_test)
    RFR.fit(X_train)
    return mean_absolute_error(y_train,RFR.predict(X_train)),mean_absolute_error(y_test,RFR.predict(X_test))
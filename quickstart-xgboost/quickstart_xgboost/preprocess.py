def preprocess_telco(df):

    df.drop(columns=['Customer ID', 'Country', 'State', 'City', 'Quarter', 'Customer Status','Churn Score' ], axis = 1, inplace=True, ) # Dropping columns wiht too many category

    df['Offer'] = df['Offer'].fillna('No Offer') # Filling missing values

    df['Internet Type'] = df['Internet Type'].fillna('No Internet') # Filling missing values

  # Binary encoding
    binary_cols = [
        'Gender',
        'Under 30',
        'Senior Citizen',
        'Married',
        'Dependents',
        'Referred a Friend',
        'Phone Service',
        'Multiple Lines',
        'Internet Service',
        'Online Security',
        'Online Backup',
        'Device Protection Plan',
        'Premium Tech Support',
        'Streaming TV',
        'Streaming Movies',
        'Streaming Music',
        'Unlimited Data',
        'Paperless Billing',
        'Churn Label'
        ]

  # Binary encoding Yes/no columns
    yes_no_cols = [
        'Under 30',
        'Senior Citizen',
        'Married',
        'Dependents',
        'Referred a Friend',
        'Phone Service',
        'Multiple Lines',
        'Internet Service',
        'Online Security',
        'Online Backup',
        'Device Protection Plan',
        'Premium Tech Support',
        'Streaming TV',
        'Streaming Movies',
        'Streaming Music',
        'Unlimited Data',
        'Paperless Billing',
        'Churn Label']

    df['Gender'] = df['Gender'].map({  # Encoding Gender columns
    'Male': 0,
    'Female': 1})
    
    one_hot_cols = [
        'Offer',
        'Internet Type',
        'Contract',
        'Payment Method']

    for col in binary_cols:  # Checking
        print(col) 
        print(df[col].unique())
        
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df.drop(
        columns=['Churn Category', 'Churn Reason'],
        axis=1,
        inplace=True )  # This is added after observing corr matrix

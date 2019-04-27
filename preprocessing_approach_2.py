#dataset = reviews_preprocess.common_dataset.drop_duplicates()

def preprocessDataframe(r_dataset):
  dataset = r_dataset.copy(deep=True)
  le = preprocessing.LabelEncoder()
  
  #Columns App, Category, Genres directly transformed
  #Variable Type: Nominal
  #Need OneHot
  
  #dataset['App'] = le.fit_transform(dataset['App'])
  i = dataset[dataset['Category'] == '1.9'].index
  dataset = dataset.drop(i)
  dataset['Category'] = le.fit_transform(dataset['Category'])
  dataset['Genres'] = le.fit_transform(dataset['Genres'])
  
  #Column: Size
  #All done. Imputed with Mean
  #Variable Type = Ratio
  def convertSizes(s):
    try:
      if s[-1] == 'M':
        return float(s[:-1]) * 1e6
      if s[-1] == 'k':
        return float(s[:-1]) * 1e3
      else:
        return 0.0
    except TypeError:
      print(s)
      
  
  new_size_col = dataset['Size'].apply(convertSizes)
  average_size = np.mean(new_size_col)
  dataset['Size'] = new_size_col.apply(lambda x: average_size if x == 0 else x)
  
  #Column: Type
  #Removed rows with nan vals
  #Variable Type: Nominal
  #Need OneHot
  nan_indices = np.where(dataset['Type'].isna().values == True)[0]
  dataset = dataset.drop(nan_indices)
  dataset.index = range(len(dataset))
  dataset['Type'] = le.fit_transform(dataset['Type'])
  
  
  #Column: Content Rating
  #Removed rows with nan vals
  #Variable Type: Nominal
  #Need OneHot
  nan_indices = np.where(dataset['Content Rating'].isna().values == True)[0]
  dataset = dataset.drop(nan_indices)
  dataset['Content Rating'] = le.fit_transform(dataset['Content Rating'])
  dataset.index = range(len(dataset))
  
  
  dataset['Rating'] = dataset['Rating'].fillna(dataset['Rating'].median())
  
  
  #Column: Installs
  #Used a custom mapper to assign categories
  #Variable Type = Ordinal
  #All done
  def categorizeInstalls():
    hmap = {}
    categories = []
    for category in dataset['Installs'].unique():
      raw = category
      raw = raw.replace(",", "")
      raw = raw.replace("+", "")
      raw = raw.replace(" ", "")
      categories.append((int(raw), category))
  
    categories.sort(key = lambda x: x[0])
    
    for i, category in enumerate(categories):
      hmap[category[1]] = i
    
    return hmap
  
  categoryMap = categorizeInstalls()
  
  def convertInstalls(s):
    return categoryMap[s]
  
  new_installs_col = dataset['Installs'].apply(convertInstalls)
  dataset['Installs'] = new_installs_col
  
  
  #Column: Price
  #Converted to float, removed $ sign
  #Variable Type: Ratio
  def convertPrices(s):
    if s[0] == "$":
      return float(s[1:])
    
    return 0.0
  
  new_prices_col = dataset['Price'].apply(convertPrices)
  dataset['Price'] = new_prices_col
  
  
  #Column: Last Updated
  #Convert to epoch time
  #Variable Type: Interval
  def convertLastUpdated(s):
    return datetime.datetime.strptime(s, "%B %d, %Y").timestamp()
  
  new_lastupdated_col = dataset['Last Updated'].apply(convertLastUpdated)
  dataset['Last Updated'] = new_lastupdated_col
  
  
  #Column: Current Ver
  #Split version among multiple columns
  """max_dots = 0
  for v in dataset['Current Ver']:
    max_dots = max(max_dots, str(v).count("."))
  
  for i in range(max_dots + 1):
    dataset["Current Ver " + str(i + 1)] = [0] * len(dataset)
  
  for i in range(len(dataset)):
    print(i)
    cur_v = str(dataset["Current Ver"][i])
    invalid = False
    for c in cur_v:
      if not (c.isdigit() or c == "."):
        invalid = True
    if invalid:
      continue
        
    split_v = cur_v.split(".")
    for j in range(len(split_v)):
      if split_v[j] != "":
        dataset["Current Ver " + str(j + 1)][i] = int(split_v[j])"""
  
  
  #Column: Android Ver
  #Split ver into multiple columns
  
  """max_dots = 0
  for v in dataset['Android Ver']:
    max_dots = max(max_dots, str(v).count("."))
  
  for i in range(max_dots + 1):
    dataset["Android Ver " + str(i + 1)] = [0] * len(dataset)
  
  for i in range(len(dataset)):
    print(i)
    cur_v = str(dataset["Android Ver"][i])
    if cur_v == "Varies with device" or cur_v == 'nan':
      continue
    cur_v = cur_v.replace(" and up", "")
    invalid = False
    for c in cur_v:
      if not (c.isdigit() or c == "."):
        invalid = True
    if invalid:
      continue
    split_v = cur_v.split(".")
    for j in range(len(split_v)):
      if split_v[j] != "":
        dataset["Android Ver " + str(j + 1)][i] = int(split_v[j])"""
  
  dataset.to_csv('preprocessed.csv')
  dataset = dataset.drop(['App', 'Current Ver', 'Android Ver'], axis=1)
  cr_f = str(sorted(dataset['Content Rating'].unique())[0])
  tp_f = str(sorted(dataset['Type'].unique())[0])
  ct_f = str(sorted(dataset['Category'].unique())[0])
  gn_f = str(sorted(dataset['Genres'].unique())[0])
  dataset = pd.get_dummies(dataset, columns=['Content Rating', 'Type', 'Category', 'Genres'])
  #print(dataset.columns)
  dataset = dataset.drop(['Content Rating_' + cr_f, 'Type_' + tp_f, 'Category_' + ct_f, 'Genres_' + gn_f], axis=1)

  return dataset

def getNumpyXy(dataset, y_column_name):
  y = dataset[y_column_name].values
  rating_index = list(dataset.columns).index(y_column_name)
  selector = [i for i in range(len(dataset.columns)) if i != rating_index]
  
  X = dataset.iloc[:, selector].values.astype(np.float64)
  
  return X, y
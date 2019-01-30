# Create empty matrix for co-occurence between pokemons
wins = pd.DataFrame(index=pokemon_df.index, columns=pokemon_df.index, data=[]).fillna(0)

for row in combats_df.values:
    
    winner = row[2]
    other = row[1] if row[1] != row[2] else row[0]
    
    wins.loc[winner, other] += 1
    
# Compute G from the wins matrix
G = np.where(wins - wins.transpose() > 0, 1, 0)

# Compute A from G
A = G + G @ G

# Aggregate sums of each line into dataframe
dominance_df = pd.DataFrame(index=pokemon_df.index, columns=['dominance'], data=np.apply_along_axis(np.sum, axis=1, arr=A))


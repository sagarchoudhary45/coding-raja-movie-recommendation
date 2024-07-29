import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# movie data
data = {
    "movieId": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)", "Waiting to Exhale (1995)", "Father of the Bride Part II (1995)", "Heat (1995)", "Sabrina (1995)", "Tom and Huck (1995)", "Sudden Death (1995)", "GoldenEye (1995)", "American President, The (1995)", "Dracula: Dead and Loving It (1995)", "Balto (1995)", "Nixon (1995)", "Cutthroat Island (1995)", "Casino (1995)", "Sense and Sensibility (1995)", "Four Rooms (1995)", "Ace Ventura: When Nature Calls (1995)", "Money Train (1995)", "Get Shorty (1995)", "Copycat (1995)", "Assassins (1995)", "Powder (1995)", "Leaving Las Vegas (1995)", "Othello (1995)", "Now and Then (1995)"],
    "genres": ["Adventure|Animation|Children|Comedy|Fantasy", "Adventure|Children|Fantasy", "Comedy|Romance", "Comedy|Drama|Romance", "Comedy", "Action|Crime|Thriller", "Comedy|Romance", "Adventure|Children", "Action", "Action|Adventure|Thriller", "Comedy|Drama|Romance", "Comedy|Horror", "Adventure|Animation|Children", "Drama", "Action|Adventure|Romance", "Crime|Drama", "Drama|Romance", "Comedy", "Comedy", "Action|Comedy|Crime|Drama|Thriller", "Comedy|Crime|Thriller", "Crime|Drama|Horror|Mystery|Thriller", "Action|Crime|Thriller", "Drama|Sci-Fi", "Drama|Romance", "Drama", "Children|Drama"]
}

movies_df = pd.DataFrame(data)
movies_df['genres'] = movies_df['genres'].str.split('|')
movies_df['genres_str'] = movies_df['genres'].apply(lambda x: ' '.join(x))

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(' '))
genre_matrix = vectorizer.fit_transform(movies_df['genres_str'])

cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = movies_df[movies_df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

print(get_recommendations('Toy Story (1995)'))

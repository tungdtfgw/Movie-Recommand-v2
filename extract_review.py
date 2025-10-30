import pandas as pd
import numpy as np


def merge_rotten_ml(
        rotten_movies_path='data/rotten_movies.csv',
        ml_data_path='data/ml-100k/u.data',
        output_path='output/movie_review_filtered.csv'):
    """
    Tạo file movie_review_filtered.csv từ dữ liệu Rotten Tomatoes và ML-100k.
    
    Args:
        rotten_movies_path: Đường dẫn đến file rotten_movies.csv
        ml100k_data_path: Đường dẫn đến file u.data của ML-100k
        output_path: Đường dẫn file output (mặc định: output/movie_review_filtered.csv)
    
    Returns:
        DataFrame chứa thông tin phim đã được filter
    """
    rotten_movies = _load_rotten_movies(rotten_movies_path)
    rotten_reviews = _load_rotten_reviews()
    movie_review_stats = _create_movie_review_stats(rotten_reviews, rotten_movies)
    movie_review_stats = _filter_by_ml100k(movie_review_stats, ml_data_path)
    movie_review_stats = movie_review_stats.drop(columns=['id'])
    movie_review_stats.to_csv(output_path, index=False)
    
    return movie_review_stats


def _load_rotten_movies(rotten_movies_path):
    """
    Đọc và loại bỏ các bản ghi trùng lặp từ file rotten_movies.csv.
    
    Args:
        rotten_movies_path: Đường dẫn đến file rotten_movies.csv
    
    Returns:
        DataFrame chứa danh sách phim không trùng lặp
    """
    rotten_movies = pd.read_csv(rotten_movies_path)
    rotten_movies = rotten_movies.drop_duplicates(subset=['id'])
    return rotten_movies


def _load_rotten_reviews():
    """
    Đọc và loại bỏ các review trùng lặp từ file rotten_reviews.csv.
    
    Returns:
        DataFrame chứa danh sách review không trùng lặp
    """
    rotten_reviews = pd.read_csv('data/rotten_reviews.csv')
    rotten_reviews = rotten_reviews.drop_duplicates(subset=['id', 'reviewId', 'creationDate'])
    return rotten_reviews


def _create_movie_review_stats(rotten_reviews, rotten_movies):
    """
    Tạo thống kê review cho mỗi phim từ dữ liệu Rotten Tomatoes.
    
    Args:
        rotten_reviews: DataFrame chứa các review
        rotten_movies: DataFrame chứa thông tin phim
    
    Returns:
        DataFrame chứa thống kê review theo phim
    """
    rotten_reviews_valid = rotten_reviews[
        rotten_reviews['reviewText'].notna() & 
        (rotten_reviews['reviewText'] != '')
    ].copy()
    
    movie_review_stats = rotten_reviews_valid.groupby('id').agg(
        n_reviews=('reviewId', 'count'),
        n_top_critic_reviews=('isTopCritic', 'sum'),
        all_reviews=('reviewText', lambda x: '|'.join([str(text) for text in x]))
    ).reset_index()
    
    movie_review_stats['n_other_reviews'] = movie_review_stats['n_reviews'] - movie_review_stats['n_top_critic_reviews']
    movie_review_stats = movie_review_stats.merge(rotten_movies[['id', 'title']], how='left', on='id')
    movie_review_stats = movie_review_stats[movie_review_stats['title'].notna()]
    movie_review_stats = movie_review_stats[
        ['id', 'title', 'n_reviews', 'n_top_critic_reviews', 'n_other_reviews', 'all_reviews']
    ]
    
    return movie_review_stats


def _filter_by_ml100k(movie_review_stats, ml100k_data_path):
    """
    Lọc movie_review_stats để chỉ giữ lại các phim có trong ML-100k dataset.
    
    Args:
        movie_review_stats: DataFrame chứa thống kê review
        ml100k_data_path: Đường dẫn đến file u.data của ML-100k
    
    Returns:
        DataFrame đã được filter theo danh sách phim trong ML-100k
    """
    ml100_titles = __load_ml100k_titles(ml100k_data_path)
    movie_review_stats['title'] = __normalize_title(movie_review_stats['title'])
    
    common_titles = set(ml100_titles).intersection(set(movie_review_stats['title'].unique()))
    movie_review_stats = movie_review_stats[movie_review_stats['title'].isin(common_titles)].copy()
    movie_review_stats['id'] = movie_review_stats['title']
    
    return movie_review_stats


def __load_ml100k_titles(ml100k_data_path):
    """
    Đọc và trả về danh sách tên phim unique từ ML-100k dataset.
    
    Args:
        ml100k_data_path: Đường dẫn đến file u.data
    
    Returns:
        Mảng numpy chứa danh sách tên phim đã được chuẩn hóa
    """
    ml100_ratings = pd.read_csv(ml100k_data_path, sep='\t', header=None, 
                               names=['userId', 'movieId', 'rating', 'timestamp'])
    
    ml100k_item_path = ml100k_data_path.replace('u.data', 'u.item')
    ml100_movies = pd.read_csv(ml100k_item_path, sep='|', header=None, encoding='latin-1',
                              names=['movieId', 'title', 'release_date', 'video_release_date', 'imdb_url'] + 
                                    [f'genre_{i}' for i in range(19)])
    
    ml100_movies['title'] = __normalize_title(ml100_movies['title'])
    ml100_df = ml100_ratings.merge(ml100_movies[['movieId', 'title']], on='movieId', how='left')
    
    return ml100_df['title'].unique()


def __normalize_title(title_series):
    """
    Chuẩn hóa tên phim: loại bỏ năm phát hành, thay khoảng trắng bằng dấu gạch dưới, chuyển về chữ thường.
    
    Args:
        title_series: Pandas Series chứa tên phim
    
    Returns:
        Pandas Series chứa tên phim đã được chuẩn hóa
    """
    return (title_series
        .str.replace(r'(\s*|\_*)\(\d{4}\)$', '', regex=True)
        .str.replace(' ', '_')
        .str.strip()
        .str.lower()
    )

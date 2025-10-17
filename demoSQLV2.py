import sys
import os
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from util.register import set_global_state, register_llm_udf
from util.utils import *
from algos.quick_greedy import QuickGreedy
import re
import json
from typing import List, Dict, Optional

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "./"))
sys.path.insert(0, project_root)


# Create a Spark session
spark = SparkSession.builder \
    .appName("LLM SQL Test") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .getOrCreate()
spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 50000)

algo_config = "quick_greedy"

base_path = os.path.dirname(os.path.abspath(__file__))
solver_config_path = dataset_config_path = os.path.join(base_path, "solver_configs", f"{algo_config}.yaml")
solver_config = read_yaml(solver_config_path)
algo = solver_config["algorithm"]
merged_cols = [] if not solver_config.get("colmerging", True) else data_config["merged_columns"]
one_deps = []
default_distinct_value_threshold = 0.0



# beer_pd_df = pd.read_csv('./datasets/beer.csv')
# beer_pd_df = clean_df(beer_pd_df)
# beer_pd_df = prepend_col_name(beer_pd_df)
# beer_df = spark.createDataFrame(beer_pd_df).repartition(1)
# beer_df.createOrReplaceTempView("movies")


movies_pd_df = pd.read_csv('./datasets/movies.csv')
movies_pd_df = clean_df(movies_pd_df)
movies_pd_df = prepend_col_name(movies_pd_df)
movies_pd_df = movies_pd_df.sample(frac=1, random_state=42)
movies_df = spark.createDataFrame(movies_pd_df).repartition(1)
movies_df.createOrReplaceTempView("movies")


# bird_pd_df =  pd.read_csv('./datasets/BIRD.csv')
# bird_pd_df = clean_df(bird_pd_df)
# bird_pd_df = prepend_col_name(bird_pd_df)
#bird_pd_df = bird_pd_df.sample(frac=1, random_state=42)
# bird_pd_df = spark.createDataFrame(bird_pd_df).repartition(1)
# bird_pd_df.createOrReplaceTempView("bird")

# pd_pd_df =  pd.read_csv('./datasets/products.csv')
# pd_pd_df = clean_df(pd_pd_df)
# pd_pd_df = prepend_col_name(pd_pd_df)
# pd_pd_df = pd_pd_df.sample(frac=1, random_state=42)
# pd_pd_df = spark.createDataFrame(pd_pd_df).repartition(1)
# pd_pd_df.createOrReplaceTempView("products")

# pdmx_pd_df =  pd.read_csv('./datasets/PDMX.csv')
# pdmx_pd_df = clean_df(pdmx_pd_df)
# pdmx_pd_df = prepend_col_name(pdmx_pd_df)
# pdmx_pd_df = pdmx_pd_df.sample(frac=1, random_state=42)
# pdmx_pd_df = spark.createDataFrame(pdmx_pd_df).repartition(1)
# pdmx_pd_df.createOrReplaceTempView("pdmx")


# Show available tables
print("test.py ----- Tables in Spark:")
spark.sql("SHOW TABLES").show()

# Set the global state for LLM functions and register the UDF
set_global_state(spark, "movies")
register_llm_udf()

"""
{
    "name": "QueryExample",
    "sql":  ("SELECT LLM(CONCAT('Summarize the review based on the following context:', ' {review_content}', '{critic_name}', '{publisher_name}', '{review_date}', ' {review_type}', '{review_score}', ' {top_critic}', '{rotten_tomatoes_link}'), r.review_content, r.critic_name, r.publisher_name, r.review_date, r.review_type, r.review_score, r.top_critic, r.rotten_tomatoes_link) AS review_summary, r.rotten_tomatoes_link FROM reviews r;")
}


{
    "name": "Query1",
    "sql": ("SELECT LLM('Recommend movies for the user based on {movie_info} and {review_content}', "
            "m.movie_info, r.review_content) AS recommendation "
            "FROM reviews r JOIN movies m ON r.rotten_tomatoes_link = m.rotten_tomatoes_link")
},
{
    "name": "TEST",
    "sql": ("SELECT LLM('Analyze whether this movie would be suitable for kids based on {movie_info} and {review_content}, only answer Yes or No', "
            "m.movie_info, r.review_content) AS recommendation "
            "FROM reviews r JOIN movies m ON r.rotten_tomatoes_link = m.rotten_tomatoes_link")
},
{
    "name": "Query2",
    "sql": ("SELECT m.movie_title "
            "FROM Movies m JOIN Reviews r ON r.rotten_tomatoes_link = m.rotten_tomatoes_link "
            "WHERE LLM('Analyze whether this movie would be suitable for kids based on {movie_info} and {review_content}, only answer Yes or No', "
            "m.movie_info, r.review_content) == 'Yes' AND r.review_type == 'Fresh'")
},
{
    "name": "Query3",
    "sql": ("SELECT AVG(LLM('Rate a satisfaction score between 0 (bad) and 5 (good) based on {review_content} and {movie_info}:, only return the score ', "
            "review_content, movie_info)) as AverageScore "
            "FROM reviews r JOIN movies m ON r.rotten_tomatoes_link = m.rotten_tomatoes_link "
            "GROUP BY m.movie_title")
},
{
    "name": "Query4",
    "sql": ("SELECT LLM('Recommend movies for the user based on {movie_info} and {review_content}', "
            "m.movie_info, r.review_content) AS recommendations "
            "FROM Movies m JOIN Reviews r ON r.rotten_tomatoes_link = m.rotten_tomatoes_link "
            "WHERE LLM('Analyze whether this movie would be suitable for kids based on {movie_info} and {review_content}, only answer Yes or No', "
            "m.movie_info, r.review_content) == 'Yes' AND LLM('Analyze whether this movie would be fresh based on {movie_info} and {review_content}, only answer Fresh or NonFresh', "
            "m.movie_info, r.review_content) == 'Fresh'")
},
{
    "name": "ALL-COLUMN",
    "sql":  "SELECT m.rotten_tomatoes_link AS movie_rotten_tomatoes_link, m.movie_title, m.movie_info, m.critics_consensus, m.content_rating, m.genres, m.directors, m.authors, m.actors, m.runtime, m.production_company, m.tomatometer_status, m.tomatometer_rating, m.tomatometer_count, m.audience_status, m.audience_rating, m.audience_count, m.tomatometer_top_critics_count, m.tomatometer_fresh_critics_count, m.tomatometer_rotten_critics_count, r.rotten_tomatoes_link AS review_rotten_tomatoes_link, r.critic_name, r.top_critic, r.publisher_name, r.review_type, r.review_score, r.review_content, LLM( 'Analyze movie details: Title: {movie_title}, Info: {movie_info}, Critics Consensus: {critics_consensus}, Content Rating: {content_rating}, Genres: {genres}, Directors: {directors}, Authors: {authors}, Actors: {actors}, Runtime: {runtime}, Production Company: {production_company}, Tomatometer Status: {tomatometer_status}, Tomatometer Rating: {tomatometer_rating}, Tomatometer Count: {tomatometer_count}, Audience Status: {audience_status}, Audience Rating: {audience_rating}, Audience Count: {audience_count}, Top Critics Count: {tomatometer_top_critics_count}, Fresh Critics Count: {tomatometer_fresh_critics_count}, Rotten Critics Count: {tomatometer_rotten_critics_count}', m.movie_title, m.movie_info, m.critics_consensus, m.content_rating, m.genres, m.directors, m.authors, m.actors, m.runtime, m.production_company, m.tomatometer_status, m.tomatometer_rating, m.tomatometer_count, m.audience_status, m.audience_rating, m.audience_count, m.tomatometer_top_critics_count, m.tomatometer_fresh_critics_count, m.tomatometer_rotten_critics_count ) AS movie_analysis, LLM( 'Analyze review details: Critic Name: {critic_name}, Top Critic: {top_critic}, Publisher: {publisher_name}, Review Type: {review_type}, Review Score: {review_score},  Review Content: {review_content}', r.critic_name, r.top_critic, r.publisher_name, r.review_type, r.review_score, r.review_content ) AS review_analysis FROM Movies m JOIN Reviews r ON m.rotten_tomatoes_link = r.rotten_tomatoes_link;"
}

{
    "name": "MULTI-FILTER-1",
    "sql": "WITH kid_friendly AS ( SELECT /*+ NO_REORDER */ m.movie_title, m.movie_info, r.review_content, r.review_type, r.rotten_tomatoes_link FROM Movies m JOIN Reviews r ON m.rotten_tomatoes_link = r.rotten_tomatoes_link WHERE LLM('Is this movie kid-friendly? Use {movie_info} and {review_content}. Answer Yes/No', m.movie_info, r.review_content) = 'Yes' ), violence_checked AS ( SELECT /*+ NO_REORDER */ * FROM kid_friendly WHERE LLM('Does {review_content} indicate minimal violence? Answer Yes/No', review_content) = 'Yes' ) SELECT /*+ NO_REORDER */ movie_title FROM violence_checked WHERE review_type = 'Fresh';"
},
{
    "name": "MULTI-FILTER-2",
    "sql": "WITH kid_friendly AS ( SELECT /*+ NO_REORDER */ m.movie_title, m.movie_info, r.review_content, r.review_type, r.rotten_tomatoes_link FROM Movies m JOIN Reviews r ON m.rotten_tomatoes_link = r.rotten_tomatoes_link WHERE LLM('Is this movie kid-friendly? Use {movie_info} and {review_content}. Answer Yes/No', m.movie_info, r.review_content) = 'Yes' ), fresh_filtered AS ( SELECT /*+ NO_REORDER */ * FROM kid_friendly WHERE review_type = 'Fresh' ) SELECT /*+ NO_REORDER */ movie_title FROM fresh_filtered WHERE LLM('Does {review_content} indicate minimal violence? Answer Yes/No', review_content) = 'Yes';",
},
{
    "name": "MULTI-FILTER-3",
    "sql": "WITH fresh_reviews AS ( SELECT /*+ NO_REORDER */ m.movie_title, m.movie_info, r.review_content, r.review_type, r.rotten_tomatoes_link FROM Movies m JOIN Reviews r ON m.rotten_tomatoes_link = r.rotten_tomatoes_link WHERE review_type = 'Fresh' ), kid_friendly AS ( SELECT /*+ NO_REORDER */ * FROM fresh_reviews WHERE LLM('Is this movie kid-friendly? Use {movie_info} and {review_content}. Answer Yes/No', movie_info, review_content) = 'Yes' ) SELECT /*+ NO_REORDER */ movie_title FROM kid_friendly WHERE LLM('Does {review_content} indicate minimal violence? Answer Yes/No', review_content) = 'Yes';",
},
{
    "name": "MULTI-FILTER-4",
    "sql": "WITH fresh_reviews AS ( SELECT /*+ NO_REORDER */ m.movie_title, m.movie_info, r.review_content, r.review_type, r.rotten_tomatoes_link FROM Movies m JOIN Reviews r ON m.rotten_tomatoes_link = r.rotten_tomatoes_link WHERE review_type = 'Fresh' ), violence_checked AS ( SELECT /*+ NO_REORDER */ * FROM fresh_reviews WHERE LLM('Does {review_content} indicate minimal violence? Answer Yes/No', review_content) = 'Yes' ), kid_friendly AS ( SELECT /*+ NO_REORDER */ * FROM violence_checked WHERE LLM('Is this movie kid-friendly? Use {movie_info} and {review_content}. Answer Yes/No', movie_info, review_content) = 'Yes' ) SELECT /*+ NO_REORDER */ movie_title FROM kid_friendly;"
}
{
    "name": "movies",
    "sql": ("SELECT LLM('Given information including movie descriptions and a critic reviews for movies with a positive sentiment, summarize the good qualities in this movie that led to a favorable rating. {reviewcontent},{topcritic},{movieinfo},{movietitle},{genres}', movies.reviewcontent, movies.topcritic, movies.movieinfo, movies.movietitle,movies.genres) from movies")
}
{
    "name": "bird",
    "sql": ("SELECT LLM('Given the following fields related to posts in an online codebase community {Text}, {PostId}, {PostDate}, {Body} , summarize how the comment Text related to the post Body', bird.Text, bird.PostId, bird.PostDate, bird.Body) from bird ")
},
{
    "name": "products",
    "sql": ("SELECT LLM('Given the following fields related to amazon products {text},{description},{parent_asin},{review_title},{verified_purchase},{rating},{product_title},{id}, summarize the product, then answer whether the product description is consistent with the quality expressed in the review.', products.text, products.description, products.parent_asin, products.review_title, products.verified_purchase, products.rating, products.product_title, products.id) from products")
}
{
    "name": "PDMX",
    "sql": ("SELECT LLM('Given the following fields {path}, {metadata}, {hasmetadata}, {version}, {isuserpro}, {isuserpublisher}, {isuserstaff}, {haspaywall}, {israted}, {isofficial}, {isoriginal}, {isdraft}, {hascustomaudio}, {hascustomvideo}, {ncomments}, {nfavorites}, {nviews}, {nratings}, {rating}, {license}, {licenseurl}, {genres}, {groups}, {tags}, {songname}, {title}, {subtitle}, {artistname}, {composername}, {publisher}, {complexity}, {ntracks}, {tracks}, {songlength}, {songlengthseconds}, {songlengthbars}, {songlengthbeats}, {nnotes}, {notesperbar}, {nannotations}, {hasannotations}, {nlyrics}, {haslyrics}, {ntokens}, {pitchclassentropy}, {scaleconsistency}, {grooveconsistency}, {bestpath}, {isbestpath}, {bestarrangement}, {isbestarrangement}, {bestuniquearrangement}, {isbestuniquearrangement}, {subsetall}, {subsetrated}, {subsetdeduplicated}, {subsetrateddeduplicated}, provide an overview on the music type, and analyze the given scores. Give exactly 50 words of summary.', pdmx.path,pdmx.metadata,pdmx.hasmetadata,pdmx.version,pdmx.isuserpro,pdmx.isuserpublisher,pdmx.isuserstaff,pdmx.haspaywall,pdmx.israted,pdmx.isofficial,pdmx.isoriginal,pdmx.isdraft,pdmx.hascustomaudio,pdmx.hascustomvideo,pdmx.ncomments,pdmx.nfavorites,pdmx.nviews,pdmx.nratings,pdmx.rating,pdmx.license,pdmx.licenseurl,pdmx.genres,pdmx.groups,pdmx.tags,pdmx.songname,pdmx.title,pdmx.subtitle,pdmx.artistname,pdmx.composername,pdmx.publisher,pdmx.complexity,pdmx.ntracks,pdmx.tracks,pdmx.songlength,pdmx.songlengthseconds,pdmx.songlengthbars,pdmx.songlengthbeats,pdmx.nnotes,pdmx.notesperbar,pdmx.nannotations,pdmx.hasannotations,pdmx.nlyrics,pdmx.haslyrics,pdmx.ntokens,pdmx.pitchclassentropy,pdmx.scaleconsistency,pdmx.grooveconsistency,pdmx.bestpath,pdmx.isbestpath,pdmx.bestarrangement,pdmx.isbestarrangement,pdmx.bestuniquearrangement,pdmx.isbestuniquearrangement,pdmx.subsetall,pdmx.subsetrated,pdmx.subsetdeduplicated,pdmx.subsetrateddeduplicated) FROM pdmx")
},
{
    "name": "products",
    "sql": ("SELECT LLM('Given the following fields related to amazon products {text},{description},{parent_asin},{review_title},{verified_purchase},{rating},{product_title},{id}, summarize the product, then answer whether the product description is consistent with the quality expressed in the review.', products.text, products.description, products.parent_asin, products.review_title, products.verified_purchase, products.rating, products.product_title, products.id) from products")
},
{
    "name": "PDMX-FILTER",
    "sql": ("SELECT songname FROM pdmx WHERE LLM('Based on following fields {path}, {metadata}, {hasmetadata}, {version}, {isuserpro}, {isuserpublisher}, {isuserstaff}, {haspaywall}, {israted}, {isofficial}, {isoriginal}, {isdraft}, {hascustomaudio}, {hascustomvideo}, {ncomments}, {nfavorites}, {nviews}, {nratings}, {rating}, {license}, {licenseurl}, {genres}, {groups}, {tags}, {songname}, {title}, {subtitle}, {artistname}, {composername}, {publisher}, {complexity}, {ntracks}, {tracks}, {songlength}, {songlengthseconds}, {songlengthbars}, {songlengthbeats}, {nnotes}, {notesperbar}, {nannotations}, {hasannotations}, {nlyrics}, {haslyrics}, {ntokens}, {pitchclassentropy}, {scaleconsistency}, {grooveconsistency}, {bestpath}, {isbestpath}, {bestarrangement}, {isbestarrangement}, {bestuniquearrangement}, {isbestuniquearrangement}, {subsetall}, {subsetrated}, {subsetdeduplicated}, {subsetrateddeduplicated}, answer ''YES'' or ''NO'' if the song name appears to reference a specific individual. If song name is empty, then answer ''NO''. Answer only ''YES'' or ''NO'', nothing else. ', pdmx.path, pdmx.metadata, pdmx.hasmetadata, pdmx.version, pdmx.isuserpro, pdmx.isuserpublisher, pdmx.isuserstaff, pdmx.haspaywall, pdmx.israted, pdmx.isofficial, pdmx.isoriginal, pdmx.isdraft, pdmx.hascustomaudio, pdmx.hascustomvideo, pdmx.ncomments, pdmx.nfavorites, pdmx.nviews, pdmx.nratings, pdmx.rating, pdmx.license, pdmx.licenseurl, pdmx.genres, pdmx.groups, pdmx.tags, pdmx.songname, pdmx.title, pdmx.subtitle, pdmx.artistname, pdmx.composername, pdmx.publisher, pdmx.complexity, pdmx.ntracks, pdmx.tracks, pdmx.songlength, pdmx.songlengthseconds, pdmx.songlengthbars, pdmx.songlengthbeats, pdmx.nnotes, pdmx.notesperbar, pdmx.nannotations, pdmx.hasannotations, pdmx.nlyrics, pdmx.haslyrics, pdmx.ntokens, pdmx.pitchclassentropy, pdmx.scaleconsistency, pdmx.grooveconsistency, pdmx.bestpath, pdmx.isbestpath, pdmx.bestarrangement, pdmx.isbestarrangement, pdmx.bestuniquearrangement, pdmx.isbestuniquearrangement, pdmx.subsetall, pdmx.subsetrated, pdmx.subsetdeduplicated, pdmx.subsetrateddeduplicated) = 'YES' ")

}

"""
queries = [
# {
#     "name": "products",
#     "sql": ("SELECT LLM('Given the following fields related to amazon products {text},{description},{parent_asin},{review_title},{verified_purchase},{rating},{product_title},{id}, summarize the product, then answer whether the product description is consistent with the quality expressed in the review.', products.text, products.description, products.parent_asin, products.review_title, products.verified_purchase, products.rating, products.product_title, products.id) from products")
# },
{
    "name": "movies",
    "sql": ("SELECT LLM('Given information including movie descriptions and a critic reviews for movies with a positive sentiment, summarize the good qualities in this movie that led to a favorable rating. {reviewcontent},{topcritic},{movieinfo},{movietitle},{genres}', movies.reviewcontent, movies.topcritic, movies.movieinfo, movies.movietitle,movies.genres) from movies")
}
]
# List to store execution times
execution_times = []

# Execute each query separately
for query_dict in queries:
    query_name = query_dict["name"]
    query_sql = query_dict["sql"]
    
    print(f"test.py ----- Executing {query_name} ...")
    start_time = time.time()


    # Execute query and show results
    result_df = spark.sql(query_sql)
    result_df.explain()
    # Save result as a CSV (one file per query)
    output_path = f"./result_{query_name}.csv"
    result_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

    end_time = time.time()
    print("execution finish")
    elapsed_time = end_time - start_time

    execution_times.append((query_name, elapsed_time))
    print(f"test.py ----- {query_name} execution time: {elapsed_time:.2f} seconds")

# Write all execution times to a file
with open("execution_times.txt", "w") as f:
    for query_name, t in execution_times:
        f.write(f"{query_name}: {t:.2f} seconds\n")

# Cleanup
spark.stop()
print("test.py ----- success ... !----------------------------------------------------------")

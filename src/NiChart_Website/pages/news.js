import React from 'react';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import Favicons from '../components/Favicons/Favicons';
import styles from '../styles/News.module.css';
import Link from 'next/link';


const News = () => {
  const newsArticles = [
    {
      title: "NiChart v1.0 is released!",
      date: "December 22, 2023",
      summary: "You can now use NiChart to process your data!",
      link: "https://twitter.com/NiChart_AIBIL"
    },
  ];
  return (
    <div className={styles.container}>
      <Head>
        <title>NiChart | News</title>
        <Favicons />
        <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
      </Head>
      <Header />
      <div className={styles.mainContent}>
        <h1>News</h1>
        <div className={styles.newsLayout}>
          <div className={styles.newsArticles}>
            {newsArticles.map((article, index) => (
              <div key={index} className={styles.article}>
                <h2>{article.title}</h2>
                <p className={styles.date}>{article.date}</p>
                <p>{article.summary}</p>
                <Link href={article.link}><a>Read more</a></Link>
              </div>
            ))}
          </div>
          <div className={styles.twitterFeed}>
            <a class="twitter-timeline" href="https://twitter.com/NiChart_AIBIL?ref_src=twsrc%5Etfw">Tweets by NiChart_AIBIL</a>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default News;

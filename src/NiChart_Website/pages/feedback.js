import React from 'react';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import Favicons from '../components/Favicons/Favicons';
import styles from '../styles/Feedback.module.css';

const Feedback = () => {
  return (
    <div className={styles.container}>
      <Head>
        <title>NiChart | Feedback</title>
        <Favicons />
      </Head>
      <Header />

      <div className={styles.feedbackSection}>
        <h1>We Value Your Feedback</h1>
        <p>Your insights and experiences are crucial for us to continuously improve NiChart. We would greatly appreciate it if you could spare a few moments to share your thoughts and feedback. Your perspective is vital in shaping the future of NiChart.</p>
        <a href="https://docs.google.com/forms/d/e/1FAIpQLSddH_eg5RHI94Ph7KYAGibzRSVfXOKReGXbj0Z2YBfF_6c8SA/viewform?usp=sf_link" className={styles.feedbackLink}>Take our 5-minute Feedback Survey</a>

        <br></br><br></br>

        <p>Please visit the <a href="/faq">Frequently Asked Questions</a> if you have questions!</p>
      </div>

      <Footer />
    </div>
  );
};

export default Feedback;

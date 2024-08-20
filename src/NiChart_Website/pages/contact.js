import React from 'react';
import Head from 'next/head';
import Header from '../components/Layout/Header';
import Footer from '../components/Layout/Footer';
import Favicons from '../components/Favicons/Favicons';
import styles from '../styles/Contact.module.css';
import { TextField, Button, Paper } from '@mui/material';



const Contact = () => {

  
  const handleSubmit = async (e) => {
    e.preventDefault();
  
    const formData = new FormData(e.target);
    const formProps = Object.fromEntries(formData);
  
    const { name, email, subject, message } = formProps;
  
    console.log(name);
    console.log(email);
    console.log(subject);
    console.log(message);
  };
  
  return (
    <div className={styles.container}>
      <Head>
        <title>NiChart | Contact Us</title>
        <Favicons />
      </Head>
      <Header />
      
      <div>
        <div className={styles.title}>
          <h1>Contact Us</h1>
          <p>NiChart is a project that's constantly under development. For that reason, we are eager for your feedback, input, ideas and questions. Please consider filling in our <a href="https://docs.google.com/forms/d/e/1FAIpQLSddH_eg5RHI94Ph7KYAGibzRSVfXOKReGXbj0Z2YBfF_6c8SA/viewform?usp=sf_link">short survey</a> (~5 minutes). Your help is greatly appreciated!</p>
          <p>Have a question or need assistance? We're here to help!</p>
        </div>
        <div className={styles.contactContainer}>
          <Paper className={styles.contactForm}>
            {/* <form onSubmit={handleSubmit}> */}
            {/* Replace formsubmit with something more stable. 250 messages per month are free. */}
            <form action="https://formsubmit.co/7631f70684d871a83d4190ccd67c01dc"  method="POST">
              <TextField label="Name" name="name" variant="outlined" fullWidth margin="normal"/>
              <TextField label="Email" name="email" variant="outlined" fullWidth margin="normal" type='email' required/>
              <TextField label="Subject" name="subject" variant="outlined" fullWidth margin="normal" />
              <TextField label="Message" name="message" variant="outlined" fullWidth multiline rows={5} margin="normal" />
              <input type="hidden" name="_next" value="https://neuroimagingchart.com/contact/"/>
              <input type="hidden" name="_subject" value="New Contact form submission!"/>
              <input type="hidden" name="_autoresponse" value="Thank you for your message!"/>
              <input type="hidden" name="_template" value="table"/>
              <input type="hidden" name="_cc" value="nichart.aibil@gmail.com"/>
              <Button type="submit" variant="contained" color="primary">Send</Button>
            </form>
          </Paper>
          <div className={styles.googleMaps}>
            <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3058.6297914270904!2d-75.20029352253465!3d39.949669871518395!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x89c6c72677dd280d%3A0x7888b382a71d5f44!2sRichards%20Medical%20Research%20Laboratories!5e0!3m2!1sen!2sus!4v1689101746189!5m2!1sen!2sus" 
                    width="100%" 
                    height="100%" 
                    style={{ border: "solid", "minHeight": "500px"}} 
                    allow="fullscreen"
                    loading="lazy">
            </iframe>
            <p className={styles.address}> 3700 Hamilton Walk Richards Building, 7th Floor Philadelphia, PA 19104</p>
            <span className={styles.phone}>
                <a href="tel:+1-215-746-4060">215-746-4060</a>
            </span>
            <br></br>
            <span className={styles.directions}>
                <a href="https://goo.gl/maps/9SkjfpSLwHY1YFzW9">Directions</a>
            </span>
            <br></br>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default Contact;


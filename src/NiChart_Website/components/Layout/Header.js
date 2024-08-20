import React, { useState, useEffect } from 'react';
import { FaBars } from 'react-icons/fa';
import styles from '/styles/Header.module.css';
import Link from 'next/link';
import ReactGA from 'react-ga4';
import {Heading, Flex, Button} from '@aws-amplify/ui-react'
const GA_TRACKING_ID = "G-CES0G22JMD";

const Header = props => {
  const [menuOpen, setMenuOpen] = useState(false);

  const {signOut, user, ...others} = props;

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const GoToCloudPortal = props => {
    return (
    <Link href="/portal"><a className={styles.portal}>NiChart Cloud</a></Link>
    )
  }
  
  const SignoutWidget = props => {
    
    if (!user) {
      return null;
    } else {
      return (
        <Flex direction="column" justifyContent="flex-start" alignContent="right" alignItems="right">
        <p><font color="white">Signed in: <br></br> {user.attributes.email}.</font></p>
        <a onClick={signOut} className={styles.portalItem}> Log out </a>
        </Flex>
      )
    }
  }

  ReactGA.initialize(GA_TRACKING_ID);
  useEffect(() => {
    ReactGA.send("pageview");
  }, []);

  return (
    <header className={styles.header}>
      <div className={styles.logos}>
        <div className={styles.logo}>
          <Link href="https://www.med.upenn.edu/cbica/">
            <a>
              <img src="/images/Logo/upenn-logo-png-white.png" alt="UPenn Logo"/>
            </a>
          </Link>
        </div>
        <div className={styles.logo}>
          <Link href="/">
            <a>
              <img src="/images/Logo/brain_transparent_logo_cropped.png" alt="NiChart Logo - Image by Gerd Altmann from Pixabay" className={styles.logoImage} />
            </a>
          </Link>
        </div>
      </div>
      <nav>
        <div className={styles.menuIcon} onClick={toggleMenu}>
          <FaBars />
        </div>
        
        <ul className={`${styles.navList} ${menuOpen ? styles.show : ''}`}>
          <li><Link href="/"><a>Home</a></Link></li>
          <li><Link href="/about"><a>About</a></Link></li>
          <li><Link href="/components"><a>Components</a></Link></li>
          <li><Link href="/team"><a>Team</a></Link></li>
          <li><Link href="/publications"><a>Publications</a></Link></li>
          <li><Link href="/feedback"><a>Feedback</a></Link></li>
          <li><Link href="/faq"><a>FAQ</a></Link></li>
          <li><Link href="/news"><a>News</a></Link></li>
          <li><Link href="/contact"><a>Contact</a></Link></li>
          <li className={styles.divider}></li>
          <li><Link href="/quickstart"><a>Quickstart</a></Link></li>
          <li>{user? <SignoutWidget/> : <GoToCloudPortal/> }</li>
        </ul>

      </nav>
    </header>
  );
};

export default Header;

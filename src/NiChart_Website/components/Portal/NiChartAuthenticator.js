import React, { useState, useEffect, useRef } from 'react';
import { Heading, Text, Button, Link, View, CheckboxField, Flex, Authenticator, Image, useTheme, useAuthenticator, TextField } from '@aws-amplify/ui-react';
import { Auth } from '@aws-amplify/auth';
import TermsModal from '../Components/TermsModal'
import Modal from '../Components/Modal'
import { Typography } from '@mui/material';


const validateEmail = (email) => {
    return String(email)
      .toLowerCase()
      .match(
        /^(([^<>()[\]\\.,;:\s@"]+(\.[^<>()[\]\\.,;:\s@"]+)*)|.(".+"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/
      );
  };

  async function resendConfirmationCode() {
    try {
        await Auth.resendSignUp(username);
        console.log('code resent successfully');
    } catch (err) {
        console.log('error resending code: ', err);
    }
}

export const NiChartAuthenticator = props => {
    const [verificationCodeModalOpen, setVerificationCodeModalOpen] = useState(false);
    const [verifyError, setVerifyError] = useState('');

    var currentEmail = '';
    const handleVerificationCodeOpen = () => {
        setVerificationCodeModalOpen(true);
    }
    
    const handleVerificationCodeClose = () => {
        setVerificationCodeModalOpen(false);
    }

    async function handleCodeModalSubmit (e) {
        // Check success with amplify signup call,
        // close modal if success, otherwise feed error to user via textfield (verifyError)
        e.preventDefault();

        const formData = new FormData(e.target)
        const formProps = Object.fromEntries(formData);
        console.log(formData)
        const username = formData.get('email');
        const confirmationCode = formData.get('code');
        console.log(username)
        console.log(confirmationCode)
        try {
            const { isSignUpComplete, nextStep } = await Auth.confirmSignUp(
                username,
                confirmationCode
            );
            handleVerificationCodeClose();
            alert("Successfully verified! Please sign in.")
        } catch (error) {
            console.log('Error confirming signup ', error)
            var errorMessage=''
            switch (error.name) {
                case 'UserNotFoundException':
                  errorMessage = 'User not found. Check email/username.';
                  break;
                case 'NotAuthorizedException':
                  errorMessage = 'Incorrect password. Try again.';
                  break;
                case 'PasswordResetRequiredException':
                  errorMessage = 'Password reset required. Check email.';
                  break;
                case 'UserNotConfirmedException':
                  errorMessage = 'User not confirmed. Verify email.';
                  break;
                case 'CodeMismatchException':
                  errorMessage = 'Invalid confirmation code. Retry.';
                  break;
                case 'ExpiredCodeException':
                  errorMessage = 'Confirmation code expired. Resend code.';
                  break;
                case 'InvalidParameterException':
                  errorMessage = 'Invalid input. Check credentials.';
                  break;
                case 'InvalidPasswordException':
                  errorMessage = 'Invalid password. Follow policy.';
                  break;
                case 'TooManyFailedAttemptsException':
                  errorMessage = 'Too many failed attempts. Wait.';
                  break;
                case 'TooManyRequestsException':
                  errorMessage = 'Request limit reached. Wait and retry.';
                  break;
                case 'LimitExceededException':
                  errorMessage = 'User pool full. Retry later.';
                  break;
                default:
                  errorMessage = 'Unknown error. Contact nichart-devs@cbica.upenn.edu for assistance.';
              }
            setVerifyError(errorMessage)
        }
    }

    function updateCurrentEmailInput (e) { currentEmail = e.target.value; }

    async function sendNewVerificationEmail (email) {
        try {
             await Auth.resendSignUp(username);
             console.log('code resent successfully');
            } catch (err) {
                console.log('error resending code: ', err);
            }
        }

    function VerificationCodeModal() {
        return (
        <Modal 
            open={verificationCodeModalOpen}
            handleClose={handleVerificationCodeClose}
            width="50%"
            title="Verification Code"
            content=""
        >
        <View textAlign="center">
        <Text>If you received a verification code but navigated away from the page, please enter your email address and code here to verify your account.</Text>
        <form class="form-example" onSubmit={handleCodeModalSubmit}>
        <div class="form-example">
            <label for="email">Enter your email address: </label>
            <input oninput={updateCurrentEmailInput} type="email" name="email" id="email" required autofocus/>
        </div>
        <div class="form-example">
            <label for="code">Enter the code you received in your verification email: </label>
            <input type="number" name="code" id="code" required />
        </div>
        <div class="form-example">
            <input type="submit" value="Verify" />
        </div>
        </form>
        <Text color="red.100">{verifyError}</Text>
        <Text>if your code has expired, enter your email and then click 
            <Button
                fontWeight="normal"
                onClick={sendNewVerificationEmail}
                size="small"
                variation="link"
            >here</Button> to re-send your verification code. The email may take a few minutes to arrive.</Text>
        </View>
        </Modal>
        )
    }

    // TermsModal stuff
    const [termsModalOpen, setTermsModalOpen] = useState(false);
    const [isCheckboxChecked, setIsCheckboxChecked] = useState(false);
    const handleCheckboxChange = (event) => {
        if (!termsModalOpen && !isCheckboxChecked) {
            setTermsModalOpen(true);
        }
        else if (!termsModalOpen && isCheckboxChecked){
            setIsCheckboxChecked(false)
        }
    };

    const handleTermsModalClose = () => {
        setTermsModalOpen(false);
    };

    const handleBottomReached = (reached) => {
        if (reached) {
          setIsCheckboxChecked(true);
      }
    };
    return (
        <Authenticator //{...props}
          // Default to Sign Up screen
          initialState="signIn"
          // Customize `Authenticator.SignUp.FormFields`
          components={{

              Header() {
                  const { tokens } = useTheme();
              
                  return (
                    <View textAlign="center" padding={tokens.space.large}>
                    
                      <Flex direction="row">
                      <Image
                        alt="NiChart Logo"
                        src="/images/Logo/brain_transparent_logo_cropped.png"
                      />
                      <Flex direction="column" justifyContent="space-around" alignContent="center" >
                          <Heading level={1}>NiChart Cloud Login</Heading>
                      </Flex>
                      </Flex>
                      
                    </View>
                  );
                },
            
              Footer() {
                const { tokens } = useTheme();
            
                return (
                  <View textAlign="center" padding={tokens.space.large}>
                    <Image
                      alt="UPenn Logo"
                      src="/images/Logo/upenn-logo-png.png"
                    />
                    <Text color={tokens.colors.neutral[80]}>
                      &copy; {new Date().getFullYear()}, Center for Biomedical Image Computing and Analytics (part of the University of Pennsylvania). 
                      All Rights Reserved.
                    </Text>
                    <Flex direction="row" justifyContent="space-around">
                        <Link rel="noopener noreferrer" href="https://www.upenn.edu/about/privacy-policy">Privacy Policy</Link>
                        <Text> | </Text>
                        <Link rel="noopener noreferrer" href="https://www.upenn.edu/about/disclaimer"> Disclaimer</Link>
                    </Flex>
                  </View>
                );
              },
            
              SignIn: {
                Header() {
                  const { tokens } = useTheme();
            
                  return (
                    <Flex direction="row" justifyContent="space-around">
                        <p>Click <Link href="/">here</Link> to return to the main site, <br/> or log in to continue to NiChart Cloud.</p>
                    </Flex>
                  );
                },
                Footer() {
                  const { toForgotPassword } = useAuthenticator();
            
                  return (
                    <View textAlign="center">
                      <Button
                        fontWeight="normal"
                        onClick={toForgotPassword}
                        size="small"
                        variation="link"
                      >
                        Reset Password
                      </Button>
                    </View>
                  );
                },
              },
            

              SignUp: {
                Header() {
                  const { tokens } = useTheme();

                  return (
                      <>
                      <View textAlign="center" padding={tokens.space.large}>
                          <Text>Please ensure you have fully read the statements on the <Link href="/about">About page</Link> and <Link href="https://www.upenn.edu/about/privacy-policy">the general University of Pennsylvania Privacy Policy</Link> before continuing.</Text>
                          <Text></Text>
                      </View>
                      <View textAlign="center" padding={tokens.space.large}>
                          <Text><b>Already have a verification code?</b> Click 
                          <Button
                              fontWeight="normal"
                              onClick={handleVerificationCodeOpen}
                              size="small"
                              variation="link"
                          >here</Button> to verify your account.</Text>
                          <VerificationCodeModal/>
                        </View>
                        <View textAlign="left" padding={tokens.space.large}>
                        <TermsModal
                            open={termsModalOpen}
                            handleClose={handleTermsModalClose}
                            title="Terms and Conditions"
                            onBottomReached={handleBottomReached}
                        />

                      </View>
                      </>
                  )
                },
                Footer() {
                  const {tokens } = useTheme();

                  return (
                      <></>
                  )
                },

                FormFields() {
                  const { validationErrors } = useAuthenticator();
                  const { tokens } = useTheme();
      
                  return (
                    <>
                      {/* Re-use default `Authenticator.SignUp.FormFields` */}
                      <Authenticator.SignUp.FormFields />
                      
                      <View textAlign="center" padding={tokens.space.large}>
                      <Text>Passwords must be at least 8 characters and contain at least one number, one uppercase letter, one lowercase letter, and one special character.</Text>
                      </View>

                      <TextField
                        name="custom:Organization"
                        label="Organization (optional)"
                      />

                      <TextField
                        name="custom:Role"
                        label="Role (optional)"
                      />
                      <CheckboxField
                        errorMessage={validationErrors.acknowledgement}
                        hasError={!!validationErrors.acknowledgement}
                        name="acknowledgement"
                        value="yes"
                        label="I agree with the Terms & Conditions"
                        // onChange={handleCheckboxChange}
                        // checked={isCheckboxChecked}
                      />
                    </>
                  );
                },
              },
          }}
          services={{
            async validateCustomSignUp(formData) {
              var errors = {};
              if (!formData.acknowledgement) {
                errors['acknowledgement'] = 'You must agree to the terms to continue.'
              }

              if (!validateEmail(formData.email)) {
                errors["email"] = 'Please enter a valid email address.'
                };
              return errors; 
              },
            }}
        >
         {/*({ signOut, user }) => (
            <main>
              <h1>Hello {user.username}</h1>
              <button onClick={signOut}>Sign out</button>
            </main>
         )*/}
         {props.children}
        </Authenticator>
      );

}


export default NiChartAuthenticator;
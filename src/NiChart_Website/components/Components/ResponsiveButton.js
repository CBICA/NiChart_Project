import { React, useState, useEffect } from 'react';
import { Button, Loader } from '@aws-amplify/ui-react';

// accepts some basic fields that also apply to AWS amplify buttons, but with slightly better functionality.
 

 
export const ResponsiveButton = props => {   
    
    const test_delay = ms => new Promise(res => setTimeout(res, ms)); 
    // ({onClick, variation, loadingText})
    
    const { onClick, loadingText, ...other } = props;
    
    
    //let [currentText, setCurrentText] = useState(inactiveText);
    let [isActive, setIsActive] = useState(false);
    
    async function handleResponsiveButtonClick () {
        if (isActive) {
            // super generic "be patient" message
            alert("The task you selected is already in progress. Please wait for the process to complete.")
            return;
        }
        setIsActive(true);
        //await test_delay(2000);
        await onClick();
        setIsActive(false);
    }
    
    return (
<Button loadingText={loadingText || "Running..."}
                isLoading={isActive} 
                onClick={handleResponsiveButtonClick}
                {...other}>
            {props.children}
        </Button>
    )
    
}

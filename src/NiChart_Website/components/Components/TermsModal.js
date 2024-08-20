import React, { useRef, useState, useEffect } from 'react';
import MUIModal from '@mui/material/Modal';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';

const TermsModal = ({ open, handleClose, title, content, onBottomReached }) => {
    const contentRef = useRef(null);
    const [closeButtonEnabled, setCloseButtonEnabled] = useState(false);

    const handleScroll = () => {
        const element = contentRef.current;
        if (element) {
            const atBottom = element.scrollHeight - element.scrollTop <= element.clientHeight;
            setCloseButtonEnabled(atBottom);

            if (atBottom && onBottomReached) {
                onBottomReached(true);
            }
        }
    };

    useEffect(() => {
        if (!open) {
            setCloseButtonEnabled(false);
        }
    }, [open]);

    const modalStyle = {
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        width: '80%',
        maxWidth: '600px',
        bgcolor: 'background.paper',
        border: '2px solid #000',
        boxShadow: 24,
        p: 4,
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
    };

    const contentStyle = {
        maxHeight: '300px',
        overflowY: 'auto',
        marginBottom: '16px',
    };

    const buttonStyle = {
        alignSelf: 'flex-end',
    };

    return (
        <MUIModal 
            open={open}
            onClose={handleClose}
            aria-labelledby="modal-title"
            aria-describedby="modal-description"
        >
            <Box sx={modalStyle}>
                <Typography id="modal-title" variant="h4" component="h2" sx={{ mb: 2 }}>
                    {title}
                </Typography>
                <div ref={contentRef} onScroll={handleScroll} style={contentStyle}>
                    <Typography variant="h6" gutterBottom>Please read the terms and scroll to the end of the text before closing this window!</Typography>
                    <br></br>
                    <br></br>
                    <Typography variant="h6" gutterBottom>NiChart Cloud Privacy Statement</Typography>
                    <Typography variant="body1">For convenience, NiChart is offered as a web service via NiChart Cloud, a service hosted using Amazon Web Services infrastructure. By uploading your data to NiChart, you are agreeing that you have valid, authorized access to that data and are not uploading personally-identifiable health information as defined by HIPAA. Uploaded scans are placed in a secure backend storage location in a private segment of the Amazon infrastructure  accessible via your login account. Individuals having root access to this server could also access your data for system maintenance purposes (e.g. to occasionally monitor folder size and delete data). Other than for system maintenance operations,  individuals with root access will never access, use or share your data with anyone.  Any uploaded data is retained for a maximum of 36 hours before being deleted. You may also choose to delete data immediately from the NiChart Cloud interface. By choosing to use NiChart Cloud, you agree that you understand these terms. If you wish to revoke this agreement at any time, simply discontinue using the service. You may contact us directly at <b><a href="mailto:nichart-devs@cbica.upenn.edu">nichart-devs@cbica.upenn.edu</a></b> about concerns related to the handling of your data. At your preference, you may also download the NiChart software tools for use on your own machine. Please see the components page for details.</Typography>
                    <br></br>
                    <br></br>
                    <Typography variant="h6" gutterBottom>FDA Disclaimer</Typography>
                    <Typography variant="body1">Please be advised that NiChart is a set of free software tools provided for research purposes. The statements made regarding the products have not been evaluated by the Food and Drug Administration. The efficacy of these products has not been confirmed by FDA-approved research. These products are not intended for clinical purposes. All information presented here is not meant as a substitute for or alternative to information from health care practitioners.</Typography>
                    <br></br>
                    <br></br>
                </div>
                <Button
                    disabled={!closeButtonEnabled}
                    onClick={handleClose}
                    style={buttonStyle}
                >
                    Close
                </Button>
            </Box>
        </MUIModal>
    );
};

export default TermsModal;

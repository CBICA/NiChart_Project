import React from 'react';
import MUIModal from '@mui/material/Modal';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';

const style = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: "800px",
  bgcolor: 'background.paper',
  boxShadow: 24,
  p: 4,
};

const Modal = props => {
  //({ open, handleClose, title, content })
  const { open, handleClose, title, content, ...other } = props;
  
  return (
    <MUIModal 
      open={open}
      onClose={handleClose}
      aria-labelledby="modal-title"
      aria-describedby="modal-description"
      {...other}
    >
      <Box sx={style}>
        <Typography id="modal-title" variant="h6" component="h2">
          {title}
        </Typography>
        <Typography id="modal-description" sx={{ mt: 2 }}>
          {content}
        </Typography>
        {props.children}
        <Button onClick={handleClose}>Close</Button>
      </Box>
    </MUIModal>
  );
};

export default Modal;


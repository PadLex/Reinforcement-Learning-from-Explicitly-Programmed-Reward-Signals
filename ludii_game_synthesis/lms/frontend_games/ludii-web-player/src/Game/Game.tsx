import React, { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';

import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';

import Accordion from '@mui/material/Accordion';
import AccordionActions from '@mui/material/AccordionActions';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import Modal from '@mui/material/Modal';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';





const Game: React.FC<{}> = () => {
  const { userId, game } = useParams();
  const [frameUrl, setFrameUrl] = useState('');
  const [userPlayer, setUserPlayer] = useState(-1);
  const [turnPlayer, setTurnPlayer] = useState(-1);
  const [gameOver, setGameOver] = useState(false);
  const [winners, setWinners] = useState([]);
  const [call, setCall] = useState(0);
  const [rulesSrc, setRulesSrc] = useState('');
  const [open, setOpen] = useState(false);
  
  const baseURL = process.env.NODE_ENV === "development" ? "http://localhost:3001" : "";
  
  const timeoutRef = useRef<number | null>(null);
  const inactivityPeriod = 15 * 60 * 1000;
  const agentThinkingPeriod = 1000; // How often to call back while waiting for an agent to make a move
  
  const scaleFactor = 1; // TODO figure out how to round everything

  const resetImage = () => {
    URL.revokeObjectURL(frameUrl);
    setFrameUrl('');
  };

  const handleActivity = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    timeoutRef.current = window.setTimeout(() => {
      resetImage();
    }, inactivityPeriod);
  };
  
  useEffect(() => {
    initializeGame(1)
  }, []);

  useEffect(() => {
    // Set up initial timeout when component mounts
    handleActivity();

    // Clean up on component unmount
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    // Reset timer whenever imageUrl changes
    handleActivity();
  }, [frameUrl]);
  
  const awaitAgent = () => {
    // Fetch agent's response
    window.setTimeout(() => {
      fetchFrame(-1, -1);
    }, agentThinkingPeriod)
  }

  const initializeGame = async (player: number) => {
    
    try {
      const response = await fetch(baseURL + '/api/start-game', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: userId,
          width: Math.round(1080 / scaleFactor),
          height: Math.round(760 / scaleFactor),
          player: player,
          game: game
        })
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json(); // Assuming the server sends JSON
      console.log(data)
      setFrameUrl(data.frame); // Using the base64 string directly
      setUserPlayer(player);
      setGameOver(data.gameOver)
      setTurnPlayer(data.player)
      setWinners(data.winners.split(','))
      setRulesSrc(data.rules)
  
      if (player !== 1) {
        await fetchFrame(Math.round(864 / scaleFactor), Math.round(736 / scaleFactor)); // Click the start button
        await fetchFrame(Math.round(864 / scaleFactor), Math.round(736 / scaleFactor));
        // awaitAgent();
      }
    } catch (error) {
      console.error('There was a problem with the fetch operation:', error);
    }
  };
  
  const fetchFrame = async (x: number, y: number) => {
    console.log("fetch:", x, y)
    try {
      const response = await fetch(baseURL + '/api/click', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: userId,
          x: x,
          y: y
        })
      });
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const data = await response.json(); // Assuming the server sends JSON
      setFrameUrl(data.frame); // Using the base64 string directly
      setGameOver(data.gameOver)
      setTurnPlayer(data.player)
      setWinners(data.winners.split(','))
      setCall(prevCall => prevCall + 1)
    } catch (error) {
      console.error('There was a problem with the fetch operation:', error);
    }
  };
  
  useEffect(() => {
    // console.log("Players changed", turnPlayer, userPlayer)
    // Await agent every time it's their turn
    if (turnPlayer !== userPlayer && !gameOver) {
      awaitAgent();
    }
  }, [turnPlayer, userPlayer, call]);


  const handleClick = (event: any) => {
    const imageElement = event.target;
    const offsetX = event.pageX - imageElement.offsetLeft;
    const offsetY = event.pageY - imageElement.offsetTop;
    
    if (userPlayer !== turnPlayer && !gameOver && offsetX < 750)
      return;
    
    fetchFrame(Math.round(offsetX / scaleFactor), Math.round(offsetY / scaleFactor));
    
    // awaitAgent();
  };

  return (
    <div style={{display: "flex", flexDirection: "row"}}>
      {frameUrl && <img src={frameUrl} width={1080} height={760} alt="Game Image" onClick={handleClick} style={{cursor: (userPlayer !== turnPlayer && !gameOver)? "progress" : "pointer"}}/>}
      
      <div>
        
        <Paper style={{display: "flex", flexDirection: "column", margin: "40px 0 40px 0"}}>
          <List>
            <ListItem>
              <ListItemText primary={!gameOver? "Game is in progress" : "GAME OVER: " + ((userPlayer in winners)? "YOU WON" : "YOU LOST")} />
            </ListItem>
            <Divider variant="middle" component="li" />
            
            {!gameOver? undefined : (
            <ListItem>
              <ListItemText primary={winners? "player(s) " + winners + " won the game" : "Nobody won the game"}/>)
            </ListItem>
            )}
            
            {!gameOver? undefined : <Divider variant="middle" component="li" />}
            
            
            <ListItem>
              <ListItemText primary={(turnPlayer === userPlayer)? "It's your turn to make a move" : "Waiting for player " + turnPlayer + " to move"} />
            </ListItem>
            <Divider variant="middle" component="li" />
            
            <ListItem>
              <ListItemText primary={"You are player " + userPlayer} />
            </ListItem>
            <Divider variant="middle" component="li" />
            
            <ListItem>
              <ListItemText primary={"Your User ID is: " + userId} />
            </ListItem>
            
          </List>
        </Paper>
        
        <Paper style={{display: "flex", flexDirection: "column", margin: "40px 0 40px 0"}}>
          <Button onClick={() => setOpen(true)} variant="outlined">View Rules</Button>
          <Button onClick={() => initializeGame(1)} variant="outlined">New Game as Player 1</Button>
          <Button onClick={() => initializeGame(2)} variant="outlined">New Game as Player 2</Button>
        </Paper>
      </div>
      
      
      
      <Modal
        open={open}
        onClose={() => setOpen(false)}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
      >
        <Box sx={{
          position: 'absolute' as 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          bgcolor: 'background.paper',
          border: '2px solid #000',
          boxShadow: 24,
          p: 4,
        }}>
          <Typography id="modal-modal-title" variant="h6" component="h2">
            Game Rules
          </Typography>
          {rulesSrc.endsWith(".png")?
            <img src={"../../.././rules/" + rulesSrc} alt="Game Rules"/> :
            <Typography id="modal-modal-description" sx={{ mt: 2 }}>{rulesSrc}</Typography>
          }
        </Box>
      </Modal>
      
    </div>
  );
};



export default Game;

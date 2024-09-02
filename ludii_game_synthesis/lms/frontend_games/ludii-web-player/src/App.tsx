import React from 'react';
import logo from './logo.svg';
import './App.css';
import Game from "./Game/Game";
import {BrowserRouter, Routes, Route, Link} from 'react-router-dom';

const Home = () => <h2>Home Page</h2>;
const About = () => <h2>About Page</h2>;
const NotFound = () => <h2>404 Not Found</h2>;
// "(game \"Tic-Tac-Toe\" (players 2) (equipment {(board (square 3)) (piece \"Disc\" P1) (piece \"Cross\" P2)}) (rules (play (move Add (to (sites Empty)))) (end (if (is Line 3) (result Mover Win)))))"
//(game "Tic-Tac-Toe" (players 2) (equipment {(board (square 3)) (piece "Disc" P1) (piece "Cross" P2)}) (rules (play (move Add (to (sites Empty)))) (end (if (is Line 3) (result Mover Win)))))

function App() {
  
  console.log(process.env.NODE_ENV)
  
  return (
    <BrowserRouter >
      <Routes>
        <Route path="/" element={<Home/>}/>
        <Route path="/about" element={<About/>}/>
        <Route path="/play/:userId/:game" element={<Game/>}/>
        <Route path="/*" element={<NotFound/>}/>
      </Routes>
    </BrowserRouter>
  );
}

export default App;

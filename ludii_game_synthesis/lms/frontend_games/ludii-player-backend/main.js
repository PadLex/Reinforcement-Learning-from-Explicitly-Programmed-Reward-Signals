const express = require('express');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const crypto= require("crypto");
const cors = require('cors');

const pino = require('pino');

const fileTransport = pino.transport({
  target: 'pino/file',
  options: { destination: `${__dirname}/app.log` },
});

const logger = pino(transport=fileTransport);

const predefinedGames = require("./predefined_games.json")

const app = express();
app.use(express.json()); // For parsing application/json

const PORT = 3001;
const INACTIVITY_TIMEOUT = 15 * 60 * 1000;


// Allow requests from your React application
app.use(cors({
  origin: 'http://localhost:3000'
  //   origin: 'https://ludii-web-player-front.loca.lt/'
}));

// Object to keep track of Java processes for each game session
let gameProcesses = {};

let processLocks = {};

let processActivity = {};

let gameIDs = {}


// Periodically check for inactive processes and kill them
setInterval(() => {
    const currentTime = Date.now();
    Object.keys(processActivity).forEach(userId => {
        if (currentTime - processActivity[userId] > INACTIVITY_TIMEOUT) {
            gameProcesses[userId].kill();
            console.log(`Process for user ${userId} killed due to inactivity.`);
            delete gameProcesses[userId];
            delete processLocks[userId];
            delete processActivity[userId];
        }
    });
}, INACTIVITY_TIMEOUT);


async function write(message, userId) {
    processActivity[userId] = Date.now();

    if (processLocks[userId]) {
        console.error(`Process locked: ${userId}`)
        return null;
    }

    processLocks[userId] = true;

    const process = gameProcesses[userId];
    console.log(`Writing: "${message}"`);
    process.stdin.write(message + '\n');

    try {
        const data = await new Promise((resolve, reject) => {
            process.stdout.once('data', resolve);
            process.once('error', reject);
        });

        console.log(`Game ${userId}: ${data.toString()}`);

        const items = data.toString().trim().split('|')

        console.log("Items:", items)

        return {path: items[0], player: parseInt(items[1]), gameOver: items[2] === "true", winners: items[3]}

    } catch (error) {
        throw new Error(`Error while receiving data from game ${userId}: ${error}`);
    } finally {
        processLocks[userId] = false;
    }
}

function sendImage(framePath, res) {
    console.log("path?", framePath)

    if (!framePath.endsWith(".png")) {
        return;
    }

    console.log("framePath", framePath)
    fs.readFile(framePath, (err, imageData) => {
        if (err) {
            res.status(500).send('Error reading image file');

            console.error(err)
            return;
        }
        res.writeHead(200, {'Content-Type': 'image/png'});
        res.end(imageData, 'binary');
    });
}


app.post('/api/start-game', async (req, res) => {
    try {
        const {userId, width, height, player, game} = req.body;

        if (gameProcesses[userId] === undefined) {
            gameProcesses[userId] = spawn('bash', ['./subprocess.sh']);

            const data = await new Promise((resolve, reject) => {
                gameProcesses[userId].stdout.once('data', resolve);
                gameProcesses[userId].once('error', reject);
            });

            if (!data.toString().startsWith("Ready")) {
                console.log("Failed to startup");
                return res.status(501);
            } else {
                console.log("YESS")
            }
        }

        gameIDs[userId] = crypto.randomBytes(20).toString('hex');


        const gameDescription = (predefinedGames[game]?.code || game).trim()
        const gameRules = predefinedGames[game]?.rules || ""

        console.log(game, "->", gameDescription)

        const javaProcess = gameProcesses[userId];

        // console.log("setup game", width, height, player, gameDescription, " -> ", userId)

        javaProcess.stdout.on('data', (data) => {
            console.log(`Data in from ${userId}: "${data.toString()}"`);
        });

        javaProcess.stderr.on('data', (data) => {
            console.error(`Error from game ${userId}: ${data.toString()}`);
        });

        javaProcess.on('close', (code) => {
            console.log(`Game ${userId} process exited with code ${code}`);
            delete gameProcesses[userId];
        });

        // TODO await for process to end
        // res.send({userId: userId});
        // javaProcess.stdin.write(`setup|${width}|${height}|${game}\n`);

        let response = await write(`setup|${width}|${height}|${player}|${gameDescription}`, userId);

        response.userId = userId
        response.frame = 'data:image/png;base64,' + fs.readFileSync(response.path, { encoding: 'base64' });
        response.rules = gameRules

        logger.info({endpoint: "start-game", userId: userId, width: width, height: height, userPlayer: player, game: game, turnPlayer: response.player, gameOver: response.gameOver, winners: response.winners, gameId: gameIDs[userId]})

        res.json(response);
    } catch (error) {
        logger.info({endpoint: "start-game", req: req, error: error.message})
        res.status(500).send(error.message);
    }
});

app.post('/api/click', async (req, res) => {
    try {
        const {userId, x, y} = req.body;
        const gameProcess = gameProcesses[userId];

        console.log("click ", userId)

        if (gameProcess) {
            let response = await write(`click|${x}|${y}`, userId);

            response.userId = userId
            response.frame = 'data:image/png;base64,' + fs.readFileSync(response.path, { encoding: 'base64' });

            logger.info({endpoint: "click", userId: userId, x: x, y: y, turnPlayer: response.player, gameOver: response.gameOver, winners: response.winners, gameId: gameIDs[userId]})

            res.json(response);
        } else {
            logger.info({endpoint: "click", userId: userId, x: x, y: y, error: "Process not found"})
            res.status(404).send('Process not found or has ended');
        }
    } catch (error) {
        logger.info({endpoint: "click", req: req, error: error.message})
        res.status(500).send(error.message);
    }
});


// Serve static files from the React app
app.use(express.static(path.join(__dirname, '/../ludii-web-player/build')));

// Handles any requests that don't match the ones above
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname+'/../ludii-web-player/build/index.html'));
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});

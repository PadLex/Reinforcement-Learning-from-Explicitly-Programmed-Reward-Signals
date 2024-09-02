document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('chess-board');
    const ctx = canvas.getContext('2d');

    // Adjust canvas drawing size
    const computedStyle = getComputedStyle(canvas);
    const width = parseInt(computedStyle.getPropertyValue('width'), 10);
    const height = parseInt(computedStyle.getPropertyValue('height'), 10);
    canvas.width = width;
    canvas.height = height;

    const boardSize = 11;
    const tileSize = width / boardSize; // Adjust based on canvas size
    let piece = null; // To store the piece's current position
    let svg = new Image(); // Declare here to be accessible in the click event listener


    function drawBoard(colors=['#f0d9b5', '#b58863']) {
        for (let i = 0; i < boardSize; i++) {
            for (let j = 0; j < boardSize; j++) {
                ctx.fillStyle = colors[(i + j) % 2];
                ctx.fillRect(j * tileSize, i * tileSize, tileSize, tileSize);
            }
        }
    }

    function addPiece(tileX= 0, tileY= 0, pieceName= 'jarl-black') {
        svg.onload = function() {
            // Initial position for the piece
            piece = {name: pieceName, x: tileX * tileSize, y: tileY * tileSize};
            ctx.drawImage(svg, piece.x, piece.y, tileSize, tileSize);
        };
        // Load the SVG file from the same directory
        svg.src = pieceName + '.svg';
    }

    // Function to move the piece
    canvas.addEventListener('click', function(e) {
        if (piece) {
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            // Move the piece to a new position
            piece.x = x - piece.width / 2;
            piece.y = y - piece.height / 2;
            drawBoard()
            ctx.drawImage(svg, piece.x, piece.y, tileSize, tileSize); // Redraw the piece at its new position
            console.log('Piece moved to', x, y);
        }
    });

    drawBoard();
    addPiece(5, 5);
});




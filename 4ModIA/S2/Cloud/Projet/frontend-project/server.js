import express from 'express';
import bodyParser from 'body-parser';
import { readFile } from 'fs';
import { resolve } from 'path';

const app = express();
const PORT = 3000;

app.use(bodyParser.urlencoded({ extended: true }));

app.post('/load-data', (req, res) => {
    const dataPath = req.body.dataPath;

    readFile(resolve(dataPath), 'utf8', (err, data) => {
        if (err) {
            return res.status(500).send('Erreur lors du chargement des données');
        }

        res.send(`<pre>${data}</pre>`);
    });
});

app.listen(PORT, () => {
    console.log(`Serveur en cours d'exécution sur le port ${PORT}`);
});

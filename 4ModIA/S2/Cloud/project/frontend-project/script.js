document.getElementById('uploadDataForm').addEventListener('submit', function(event) {
    event.preventDefault();
    let dataPath = document.getElementById('dataPath').value;

    fetch('/upload_data', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ dataPath: dataPath })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Erreur lors du chargement des données');
        }
        return response.json();
    })
    .then(data => {
        alert('Données chargées avec succès !');
        // Réinitialisation du formulaire ou autre action à faire après le chargement
    })
    .catch(error => {
        alert('Erreur : ' + error.message);
    });
});

document.getElementById('uploadProgramForm').addEventListener('submit', function(event) {
    event.preventDefault();
    let formData = new FormData();
    formData.append('programFile', document.getElementById('programFile').files[0]);

    fetch('/upload_program', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Erreur lors de l\'upload du programme');
        }
        return response.json();
    })
    .then(data => {
        alert('Programme uploadé avec succès !');
        // Réinitialisation du formulaire ou autre action à faire après l'upload
    })
    .catch(error => {
        alert('Erreur : ' + error.message);
    });
});

document.getElementById('executeApplicationForm').addEventListener('submit', function(event) {
    event.preventDefault();
    let program = document.getElementById('programSelect').value;
    let inputFile = document.getElementById('inputFile').value;
    let outputDir = document.getElementById('outputDir').value;

    fetch('/execute_application', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ program: program, inputFile: inputFile, outputDir: outputDir })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Erreur lors de l\'exécution de l\'application');
        }
        return response.json();
    })
    .then(data => {
        alert('Application exécutée avec succès !');
        // Gérer le téléchargement du fichier de sortie si nécessaire
    })
    .catch(error => {
        alert('Erreur : ' + error.message);
    });
});

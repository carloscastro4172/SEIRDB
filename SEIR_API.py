import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from datetime import datetime, timedelta
from flask_migrate import Migrate

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:my_password@localhost:3306/EpidemicDB'

db = SQLAlchemy(app)
ma = Marshmallow(app)

# Modelo Location
class Location(db.Model):
    __tablename__ = 'LOCATION'

    LocationID = db.Column(db.Integer, primary_key=True, nullable=False)
    Latitude = db.Column(db.Float, nullable=False)
    LocationName = db.Column(db.String(255), nullable=False)
    Longitude = db.Column(db.Float, nullable=False)


# Modelo Epidemics
class Epidemics(db.Model):
    __tablename__ = 'EPIDEMICS'

    EpidemicID = db.Column(db.Integer, primary_key=True, nullable=False)
    LocationID = db.Column(db.Integer, db.ForeignKey('LOCATION.LocationID'), nullable=False)
    NameEpidemic = db.Column(db.String(255), nullable=False)
    StartDate = db.Column(db.Date, nullable=False)
    EndDate = db.Column(db.Date, nullable=False)
    Description = db.Column(db.String(255), nullable=False)


# Modelo Simulations
class Simulations(db.Model):
    __tablename__ = 'SIMULATIONS'

    SimulationID = db.Column(db.Integer, primary_key=True, nullable=False)
    EpidemicID = db.Column(db.Integer, db.ForeignKey('EPIDEMICS.EpidemicID'), nullable=False)


# Modelo SEIRParameters
class SEIRParameters(db.Model):
    __tablename__ = 'SEIR_PARAMETERS'

    ResultID = db.Column(db.Integer, primary_key=True, nullable=False)
    SimulationID = db.Column(db.Integer, db.ForeignKey('SIMULATIONS.SimulationID'), nullable=False)
    Beta = db.Column(db.Float, nullable=False)
    Sigma = db.Column(db.Float, nullable=False)
    Gamma = db.Column(db.Float, nullable=False)
    Suceptible = db.Column(db.Integer, nullable=False)
    Exposed = db.Column(db.Integer, nullable=False)
    Infected = db.Column(db.Integer, nullable=False)
    Recovered = db.Column(db.Integer, nullable=False)
    DateSimulation = db.Column(db.Date, nullable=False)
    TimeFinal = db.Column(db.Integer, nullable=False)


# Modelo UserData
class UserData(db.Model):
    __tablename__ = 'USER_DATA'

    UserID = db.Column(db.Integer, primary_key=True, nullable=False)
    Role = db.Column(db.String(50), nullable=False)
    Email = db.Column(db.String(255), nullable=False)


class SEIRResults(db.Model):
    __tablename__ = 'SEIR_RESULTS'

    ResultID = db.Column(db.Integer, primary_key=True, nullable=False, autoincrement=True)
    
    SimulationID = db.Column(db.Integer, db.ForeignKey('SIMULATIONS.SimulationID', ondelete='CASCADE'), nullable=False)
    
    Result_Suceptible = db.Column(db.Integer, nullable=False)
    Result_Exposed = db.Column(db.Integer, nullable=False)
    Result_Infected = db.Column(db.Integer, nullable=False)
    Result_Recovered = db.Column(db.Integer, nullable=False)
    Date_Simulation = db.Column(db.Date, nullable=False)
    UserID = db.Column(db.Integer, db.ForeignKey('USER_DATA.UserID'), nullable=False)

# Crear las tablas en la base de datos
@app.before_request
def setup():
    db.create_all()
# Esquema Location
class LocationSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Location
        include_fk = True

# Esquema Epidemics
class EpidemicsSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Epidemics
        include_fk = True

# Esquema Simulations
class SimulationsSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Simulations
        include_fk = True

# Esquema SEIRParameters
class SEIRParametersSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = SEIRParameters
        include_fk = True

# Esquema UserData
class UserDataSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = UserData
        include_fk = True

# Esquema SEIRResults
class SEIRResultsSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = SEIRResults
        include_fk = True

# Endpoints

# Endpoints para UserData
from flask import request, jsonify

# Crear un nuevo usuario
@app.route('/userdata', methods=['POST'])
def create_user_data():
    data = request.get_json()
    new_user = UserData(
        UserID=data['UserID'],
        Role=data['Role'],
        Email=data['Email']
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify(UserDataSchema().dump(new_user)), 201


# Obtener todos los usuarios
@app.route('/userdata', methods=['GET'])
def get_all_user_data():
    users = UserData.query.all()
    return jsonify(UserDataSchema(many=True).dump(users)), 200


# Obtener un usuario por ID
@app.route('/userdata/<int:id>', methods=['GET'])
def get_user_data(id):
    user = UserData.query.get(id)
    if not user:
        return jsonify({"message": "User not found"}), 404
    return jsonify(UserDataSchema().dump(user)), 200


# Actualizar un usuario
@app.route('/userdata/<int:id>', methods=['PUT'])
def update_user_data(id):
    user = UserData.query.get(id)
    if not user:
        return jsonify({"message": "User not found"}), 404

    data = request.get_json()
    user.Role = data.get('Role', user.Role)
    user.Email = data.get('Email', user.Email)
    db.session.commit()
    return jsonify(UserDataSchema().dump(user)), 200


# Eliminar un usuario
@app.route('/userdata/<int:id>', methods=['DELETE'])
def delete_user_data(id):
    user = UserData.query.get(id)
    if not user:
        return jsonify({"message": "User not found"}), 404

    db.session.delete(user)
    db.session.commit()
    return jsonify({"message": "User deleted successfully"}), 200


# Endpoints para Locations

# Crear una nueva ubicación
@app.route('/location', methods=['POST'])
def create_location():
    data = request.get_json()
    new_location = Location(
        LocationID=data['LocationID'],
        Latitude=data['Latitude'],
        LocationName=data['LocationName'],
        Longitude=data['Longitude']
    )
    db.session.add(new_location)
    db.session.commit()
    return jsonify(LocationSchema().dump(new_location)), 201


# Obtener todas las ubicaciones
@app.route('/location', methods=['GET'])
def get_all_locations():
    locations = Location.query.all()
    return jsonify(LocationSchema(many=True).dump(locations)), 200


# Obtener una ubicación por ID
@app.route('/location/<int:id>', methods=['GET'])
def get_location(id):
    location = Location.query.get(id)
    if not location:
        return jsonify({"message": "Location not found"}), 404
    return jsonify(LocationSchema().dump(location)), 200


# Actualizar una ubicación
@app.route('/location/<int:id>', methods=['PUT'])
def update_location(id):
    location = Location.query.get(id)
    if not location:
        return jsonify({"message": "Location not found"}), 404

    data = request.get_json()
    location.Latitude = data.get('Latitude', location.Latitude)
    location.LocationName = data.get('LocationName', location.LocationName)
    location.Longitude = data.get('Longitude', location.Longitude)
    db.session.commit()
    return jsonify(LocationSchema().dump(location)), 200


# Eliminar una ubicación
@app.route('/location/<int:id>', methods=['DELETE'])
def delete_location(id):
    location = Location.query.get(id)
    if not location:
        return jsonify({"message": "Location not found"}), 404

    db.session.delete(location)
    db.session.commit()
    return jsonify({"message": "Location deleted successfully"}), 200


# Endpoints Epidemics
from flask import request, jsonify

# Crear una nueva epidemia
@app.route('/epidemics', methods=['POST'])
def create_epidemic():
    data = request.get_json()
    new_epidemic = Epidemics(
        EpidemicID=data['EpidemicID'],
        LocationID=data['LocationID'],
        NameEpidemic=data['NameEpidemic'],
        StartDate=data['StartDate'],
        EndDate=data['EndDate'],
        Description=data['Description']
    )
    db.session.add(new_epidemic)
    db.session.commit()
    return jsonify(EpidemicsSchema().dump(new_epidemic)), 201


# Obtener todas las epidemias
@app.route('/epidemics', methods=['GET'])
def get_all_epidemics():
    epidemics = Epidemics.query.all()
    return jsonify(EpidemicsSchema(many=True).dump(epidemics)), 200


# Obtener una epidemia por ID
@app.route('/epidemics/<int:id>', methods=['GET'])
def get_epidemic(id):
    epidemic = Epidemics.query.get(id)
    if not epidemic:
        return jsonify({"message": "Epidemic not found"}), 404
    return jsonify(EpidemicsSchema().dump(epidemic)), 200


# Actualizar una epidemia
@app.route('/epidemics/<int:id>', methods=['PUT'])
def update_epidemic(id):
    epidemic = Epidemics.query.get(id)
    if not epidemic:
        return jsonify({"message": "Epidemic not found"}), 404

    data = request.get_json()
    epidemic.LocationID = data.get('LocationID', epidemic.LocationID)
    epidemic.NameEpidemic = data.get('NameEpidemic', epidemic.NameEpidemic)
    epidemic.StartDate = data.get('StartDate', epidemic.StartDate)
    epidemic.EndDate = data.get('EndDate', epidemic.EndDate)
    epidemic.Description = data.get('Description', epidemic.Description)
    db.session.commit()
    return jsonify(EpidemicsSchema().dump(epidemic)), 200


# Eliminar una epidemia
@app.route('/epidemics/<int:id>', methods=['DELETE'])
def delete_epidemic(id):
    epidemic = Epidemics.query.get(id)
    if not epidemic:
        return jsonify({"message": "Epidemic not found"}), 404

    db.session.delete(epidemic)
    db.session.commit()
    return jsonify({"message": "Epidemic deleted successfully"}), 200

#Enpoint Simulations
from flask import request, jsonify

# Crear una nueva simulación
@app.route('/simulations', methods=['POST'])
def create_simulation():
    data = request.get_json()
    new_simulation = Simulations(
        SimulationID=data['SimulationID'],
        EpidemicID=data['EpidemicID']
    )
    db.session.add(new_simulation)
    db.session.commit()
    return jsonify(SimulationsSchema().dump(new_simulation)), 201


# Obtener todas las simulaciones
@app.route('/simulations', methods=['GET'])
def get_all_simulations():
    simulations = Simulations.query.all()
    return jsonify(SimulationsSchema(many=True).dump(simulations)), 200


# Obtener una simulación por ID
@app.route('/simulations/<int:id>', methods=['GET'])
def get_simulation(id):
    simulation = Simulations.query.get(id)
    if not simulation:
        return jsonify({"message": "Simulation not found"}), 404
    return jsonify(SimulationsSchema().dump(simulation)), 200


# Actualizar una simulación
@app.route('/simulations/<int:id>', methods=['PUT'])
def update_simulation(id):
    simulation = Simulations.query.get(id)
    if not simulation:
        return jsonify({"message": "Simulation not found"}), 404

    data = request.get_json()
    simulation.EpidemicID = data.get('EpidemicID', simulation.EpidemicID)
    db.session.commit()
    return jsonify(SimulationsSchema().dump(simulation)), 200


# Eliminar una simulación
@app.route('/simulations/<int:id>', methods=['DELETE'])
def delete_simulation(id):
    simulation = Simulations.query.get(id)
    if not simulation:
        return jsonify({"message": "Simulation not found"}), 404

    db.session.delete(simulation)
    db.session.commit()
    return jsonify({"message": "Simulation deleted successfully"}), 200

# ENDPOINTS SEIRPARAMETERS

from flask import request, jsonify

# Crear un nuevo SEIR parameter
@app.route('/seirparameters', methods=['POST'])
def create_seir_parameter():
    data = request.get_json()
    new_seir = SEIRParameters(
        ResultID=data['ResultID'],
        SimulationID=data['SimulationID'],
        Beta=data['Beta'],
        Sigma=data['Sigma'],
        Gamma=data['Gamma'],
        Suceptible=data['Suceptible'],
        Exposed=data['Exposed'],
        Infected=data['Infected'],
        Recovered=data['Recovered'],
        DateSimulation=data['DateSimulation'],
        TimeFinal=data['TimeFinal']
    )
    db.session.add(new_seir)
    db.session.commit()
    return jsonify(SEIRParametersSchema().dump(new_seir)), 201


# Obtener todos los SEIR parameters
@app.route('/seirparameters', methods=['GET'])
def get_all_seir_parameters():
    seir_parameters = SEIRParameters.query.all()
    return jsonify(SEIRParametersSchema(many=True).dump(seir_parameters)), 200


# Obtener un SEIR parameter por ID
@app.route('/seirparameters/<int:id>', methods=['GET'])
def get_seir_parameter(id):
    seir_parameter = SEIRParameters.query.get(id)
    if not seir_parameter:
        return jsonify({"message": "SEIR parameter not found"}), 404
    return jsonify(SEIRParametersSchema().dump(seir_parameter)), 200


# Actualizar un SEIR parameter
@app.route('/seirparameters/<int:id>', methods=['PUT'])
def update_seir_parameter(id):
    seir_parameter = SEIRParameters.query.get(id)
    if not seir_parameter:
        return jsonify({"message": "SEIR parameter not found"}), 404

    data = request.get_json()
    seir_parameter.SimulationID = data.get('SimulationID', seir_parameter.SimulationID)
    seir_parameter.Beta = data.get('Beta', seir_parameter.Beta)
    seir_parameter.Sigma = data.get('Sigma', seir_parameter.Sigma)
    seir_parameter.Gamma = data.get('Gamma', seir_parameter.Gamma)
    seir_parameter.Suceptible = data.get('Suceptible', seir_parameter.Suceptible)
    seir_parameter.Exposed = data.get('Exposed', seir_parameter.Exposed)
    seir_parameter.Infected = data.get('Infected', seir_parameter.Infected)
    seir_parameter.Recovered = data.get('Recovered', seir_parameter.Recovered)
    seir_parameter.DateSimulation = data.get('DateSimulation', seir_parameter.DateSimulation)
    seir_parameter.TimeFinal = data.get('TimeFinal', seir_parameter.TimeFinal)
    
    db.session.commit()
    return jsonify(SEIRParametersSchema().dump(seir_parameter)), 200


# Eliminar un SEIR parameter
@app.route('/seirparameters/<int:id>', methods=['DELETE'])
def delete_seir_parameter(id):
    seir_parameter = SEIRParameters.query.get(id)
    if not seir_parameter:
        return jsonify({"message": "SEIR parameter not found"}), 404

    db.session.delete(seir_parameter)
    db.session.commit()
    return jsonify({"message": "SEIR parameter deleted successfully"}), 200

#ENDPOINTS SEIRResults
from flask import request, jsonify

# Crear un nuevo SEIR result
@app.route('/seirresults', methods=['POST'])
def create_seir_result():
    data = request.get_json()
    new_seir_result = SEIRResults(
        ResultID=data['ResultID'],
        SimulationID=data['SimulationID'],
        Result_Suceptible=data['Result_Suceptible'],
        Result_Exposed=data['Result_Exposed'],
        Result_Infected=data['Result_Infected'],
        Result_Recovered=data['Result_Recovered'],
        Date_Simulation=data['Date_Simulation'],
        UserID=data['UserID']
    )
    db.session.add(new_seir_result)
    db.session.commit()
    return jsonify(SEIRResultsSchema().dump(new_seir_result)), 201


# Obtener todos los SEIR results
@app.route('/seirresults', methods=['GET'])
def get_all_seir_results():
    seir_results = SEIRResults.query.all()
    return jsonify(SEIRResultsSchema(many=True).dump(seir_results)), 200


# Obtener un SEIR result por ID
@app.route('/seirresults/<int:id>', methods=['GET'])
def get_seir_result(id):
    seir_result = SEIRResults.query.get(id)
    if not seir_result:
        return jsonify({"message": "SEIR result not found"}), 404
    return jsonify(SEIRResultsSchema().dump(seir_result)), 200


# Actualizar un SEIR result
@app.route('/seirresults/<int:id>', methods=['PUT'])
def update_seir_result(id):
    seir_result = SEIRResults.query.get(id)
    if not seir_result:
        return jsonify({"message": "SEIR result not found"}), 404

    data = request.get_json()
    seir_result.SimulationID = data.get('SimulationID', seir_result.SimulationID)
    seir_result.Result_Suceptible = data.get('Result_Suceptible', seir_result.Result_Suceptible)
    seir_result.Result_Exposed = data.get('Result_Exposed', seir_result.Result_Exposed)
    seir_result.Result_Infected = data.get('Result_Infected', seir_result.Result_Infected)
    seir_result.Result_Recovered = data.get('Result_Recovered', seir_result.Result_Recovered)
    seir_result.Date_Simulation = data.get('Date_Simulation', seir_result.Date_Simulation)
    seir_result.UserID = data.get('UserID', seir_result.UserID)
    
    db.session.commit()
    return jsonify(SEIRResultsSchema().dump(seir_result)), 200


# Eliminar un SEIR result
@app.route('/seirresults/<int:id>', methods=['DELETE'])
def delete_seir_result(id):
    seir_result = SEIRResults.query.get(id)
    if not seir_result:
        return jsonify({"message": "SEIR result not found"}), 404

    db.session.delete(seir_result)
    db.session.commit()
    return jsonify({"message": "SEIR result deleted successfully"}), 200


#ENDPOINT SIMULATION

from datetime import datetime, timedelta
import numpy as np
from scipy.integrate import odeint

@app.route('/simulate_seir/<int:id>', methods=['POST'])
def simulate_seir_with_saved_params(id):
    parameter = SEIRParameters.query.get(id)
    if not parameter:
        return make_response(jsonify({"message": "Parameters not found"}), 404)

    # Extraer parámetros del registro
    try:
        S0 = int(parameter.Suceptible)
        E0 = int(parameter.Exposed)
        I0 = int(parameter.Infected)
        R0 = int(parameter.Recovered)
        Beta = float(parameter.Beta)
        Sigma = float(parameter.Sigma)
        Gamma = float(parameter.Gamma)
        t_max = int(parameter.TimeFinal)  # Convertir TimeFinal a entero
    except (ValueError, TypeError):
        return make_response(jsonify({"message": "Invalid parameter format"}), 400)

    # Validar valores iniciales y tiempo
    if S0 < 0 or E0 < 0 or I0 < 0 or R0 < 0:
        return make_response(jsonify({"message": "Initial conditions must be non-negative"}), 400)
    if t_max <= 0:
        return make_response(jsonify({"message": "TimeFinal must be a positive integer"}), 400)

    # Modelo SEIR y simulación
    dt = 1
    N = S0 + E0 + I0 + R0
    if N <= 0:
        return make_response(jsonify({"message": "Population (S0 + E0 + I0 + R0) must be greater than zero"}), 400)

    alpha = Sigma
    beta = Beta
    gamma = Gamma
    t = np.arange(0, t_max, dt)

    def seir_eq(v, t, alpha, beta, gamma, N):
        S, E, I, R = v
        dS = -beta * S * I / N
        dE = beta * S * I / N - alpha * E
        dI = alpha * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]

    ini_state = [S0, E0, I0, R0]
    solution = odeint(seir_eq, ini_state, t, args=(alpha, beta, gamma, N))
    S, E, I, R = solution.T

    # Establecer la fecha de inicio de la simulación (ajusta según tus necesidades)
    start_date = datetime(2024, 1, 1)  # Fecha de inicio (ajústala según sea necesario)

    # Guardar resultados en la tabla SEIR_Results con valores enteros
    for i in range(len(t)):
        # Calcular la fecha de simulación basada en `t[i]` (número de días desde el inicio)
        date_simulation = start_date + timedelta(days=int(t[i]))

        result = SEIRResults(
            SimulationID=id,
            Result_Suceptible=int(S[i]),  # Convertimos a entero
            Result_Exposed=int(E[i]),     # Convertimos a entero
            Result_Infected=int(I[i]),    # Convertimos a entero
            Result_Recovered=int(R[i]),   # Convertimos a entero
            Date_Simulation=date_simulation.date(),  # Guardamos la fecha de simulación
            UserID=1  # Puedes ajustar este valor de acuerdo al usuario que realiza la simulación
        )
        db.session.add(result)

    db.session.commit()  # Confirmamos los cambios en la base de datos

    # Generamos la respuesta y convertimos las listas a enteros
    response = {
        "time": t.tolist(),
        "suceptible": [int(x) for x in S.tolist()],  # Convertir los valores de susceptible a enteros
        "exposed": [int(x) for x in E.tolist()],    # Convertir los valores de exposed a enteros
        "infected": [int(x) for x in I.tolist()],   # Convertir los valores de infected a enteros
        "recovered": [int(x) for x in R.tolist()]   # Convertir los valores de recovered a enteros
    }

    return jsonify(response), 200

#ENDPOINT JOIN 
@app.route('/epidemics_with_simulations', methods=['GET'])
def get_epidemics_with_simulations():
    results = db.session.query(
        Epidemics.NameEpidemic,
        Location.LocationName,
        Location.Latitude,
        Location.Longitude,
        Simulations.SimulationID
    ).join(Location, Epidemics.LocationID == Location.LocationID) \
     .outerjoin(Simulations, Epidemics.EpidemicID == Simulations.EpidemicID).all()

    response = [
        {
            "NameEpidemic": r[0],
            "LocationName": r[1],
            "Latitude": r[2],
            "Longitude": r[3],
            "SimulationID": r[4]
        } for r in results
    ]
    return jsonify(response), 200

from sqlalchemy import text

@app.route('/epidemic_peaks', methods=['GET'])
def get_epidemic_peaks():
    # Definir la consulta SQL con text(), usando DATE() para eliminar la hora de PeakDate
    query = text("""
    WITH PeakData AS (
        SELECT 
            e.EpidemicID,
            DATE(sr.Date_Simulation) AS PeakDate,  -- Usar DATE() para solo obtener la fecha
            sr.Result_Infected AS PeakInfected
        FROM 
            EPIDEMICS e
        JOIN 
            SIMULATIONS s ON e.EpidemicID = s.EpidemicID
        JOIN 
            SEIR_RESULTS sr ON s.SimulationID = sr.SimulationID
        WHERE 
            sr.Result_Infected = (
                SELECT MAX(sr_sub.Result_Infected)
                FROM SEIR_RESULTS sr_sub
                JOIN SIMULATIONS s_sub ON sr_sub.SimulationID = s_sub.SimulationID
                WHERE s_sub.EpidemicID = e.EpidemicID
            )
    )
    SELECT 
        e.NameEpidemic,
        AVG(sr.Result_Infected) AS AvgInfected,
        MAX(sr.Result_Infected) AS PeakInfected,
        pd.PeakDate
    FROM 
        EPIDEMICS e
    JOIN 
        SIMULATIONS s ON e.EpidemicID = s.EpidemicID
    JOIN 
        SEIR_RESULTS sr ON s.SimulationID = sr.SimulationID
    JOIN 
        PeakData pd ON e.EpidemicID = pd.EpidemicID
    GROUP BY 
        e.NameEpidemic, pd.PeakDate;
    """)

    # Ejecutar la consulta
    results = db.session.execute(query)

    # Convertir los resultados a un formato de lista de diccionarios
    rows = results.fetchall()
    columns = results.keys()  # Obtener los nombres de las columnas

    # Mapear los resultados a una lista de diccionarios
    response = [
        {
            "NameEpidemic": row[0],
            "AvgInfected": row[1],
            "PeakInfected": row[2],
            "PeakDate": row[3]  # Ya no incluirá la hora
        } for row in rows
    ]

    # Retornar la respuesta en formato JSON
    return jsonify(response), 200

@app.route('/epidemic_recoveries', methods=['GET'])
def get_epidemic_recoveries():
    # Definir la consulta SQL para los recuperados
    query = text("""
    WITH PeakData AS (
        SELECT 
            e.EpidemicID,
            sr.Date_Simulation AS PeakDate,
            sr.Result_Recovered AS PeakRecovered  -- Cambiar a Result_Recovered
        FROM 
            EPIDEMICS e
        JOIN 
            SIMULATIONS s ON e.EpidemicID = s.EpidemicID
        JOIN 
            SEIR_RESULTS sr ON s.SimulationID = sr.SimulationID
        WHERE 
            sr.Result_Recovered = (
                SELECT MAX(sr_sub.Result_Recovered)
                FROM SEIR_RESULTS sr_sub
                JOIN SIMULATIONS s_sub ON sr_sub.SimulationID = s_sub.SimulationID
                WHERE s_sub.EpidemicID = e.EpidemicID
            )
    )
    SELECT 
        e.NameEpidemic,
        AVG(sr.Result_Recovered) AS AvgRecovered,  -- Promedio de recuperados
        MAX(sr.Result_Recovered) AS PeakRecovered,  -- Máximo de recuperados
        pd.PeakDate
    FROM 
        EPIDEMICS e
    JOIN 
        SIMULATIONS s ON e.EpidemicID = s.EpidemicID
    JOIN 
        SEIR_RESULTS sr ON s.SimulationID = sr.SimulationID
    JOIN 
        PeakData pd ON e.EpidemicID = pd.EpidemicID
    GROUP BY 
        e.NameEpidemic, pd.PeakDate;
    """)

    # Ejecutar la consulta
    results = db.session.execute(query)

    # Convertir los resultados a un formato de lista de diccionarios
    rows = results.fetchall()

    # Procesar los resultados
    response = [
        {
            "NameEpidemic": row[0],
            "AvgRecovered": row[1],  # Promedio de recuperados
            "PeakRecovered": row[2],  # Pico de recuperados
            "PeakDate": row[3]  # Fecha del pico
        } for row in rows
    ]

    # Retornar la respuesta en formato JSON
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True)


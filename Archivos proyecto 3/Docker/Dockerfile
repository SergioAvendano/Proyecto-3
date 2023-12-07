# eyntax-docker/dockerfile:1
FROM ubuntu:22.04

#Instalar dependencias
RUN apt-get update && apt-get install -y python3 python3-pip

# Crear un directorio de trabajo
WORKDIR /app/

# Copiar archivos necesarios
COPY Dashdefinitivo.py /app/
COPY DatosP3rinal.xlsx /app/
COPY ResultadoshT.pkl /app/

# Instalar dependencias de Python
RUN pip3 install —-upgrade pip
RUN apt-get update £6 apt-get install -y python3-pip

# Instalar bibliotecas de Python
RUN pip3 install pandas pgmpy
RUN pip3 install gunicorn

RUN pip3 install dash

RUN pip3 install ucimlrepo

RUN pip3 install matplotlib

# Puerto a exponer para el tablero
EXPOSE 8050

# Comandos a ejecutar al correr el contenedor
CMD ["python3", "Dashdefinitivo.py"]
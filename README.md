# 🐍 gee_service - Servicio de Procesamiento GEE (Flask/Python)

Este directorio contiene el backend principal del Visor de NDVI Histórico, implementado con Flask. Su única responsabilidad es interactuar con la API de Google Earth Engine (GEE) para procesar datos satelitales Sentinel-2.

## ⚙️ Funcionalidad Principal

El servicio expone un único endpoint:

- **POST `/api/ndvi`**: Recibe la geometría (AOI) y la fecha, calcula el NDVI (Índice de Vegetación de Diferencia Normalizada) promedio y genera una URL de mosaico (tile URL) para superponer en el mapa del frontend (Leaflet).

## 🛠️ Configuración y Autenticación Híbrida

La aplicación está diseñada para autenticarse con GEE de forma flexible, permitiendo un desarrollo local seguro y un despliegue en producción sin subir archivos de claves privadas.

### 1. Variables de Entorno (Local)

Crea un archivo **`.env`** en este directorio (`gee_service/`) con las siguientes variables:

| Variable              | Uso                                                                                                      |
| :-------------------- | :------------------------------------------------------------------------------------------------------- |
| `EE_SERVICE_ACCOUNT`  | Correo electrónico de tu cuenta de servicio GEE.                                                         |
| `EE_PRIVATE_KEY_PATH` | Ruta al archivo JSON de tu clave privada (ej: `./private-key_pedralcg.json`). **Solo en entorno local.** |

### 2. Despliegue en Render (Producción)

Para el despliegue, **NO** subas el archivo JSON de clave privada a GitHub. En su lugar, usa la configuración de variables de entorno de Render:

| Variable (en Render)     | Contenido                                                                            |
| :----------------------- | :----------------------------------------------------------------------------------- |
| `EE_PRIVATE_KEY_CONTENT` | El contenido JSON **COMPLETO** de tu clave privada (como una única cadena de texto). |

El archivo `app.py` utiliza una lógica híbrida que da prioridad a `EE_PRIVATE_KEY_CONTENT` para producción y recurre a `EE_PRIVATE_KEY_PATH` solo en el entorno local.

## ▶️ Ejecución Local

1.  **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Asegurar la autenticación local:**

    - Verifica que tu archivo `.env` esté configurado.
    - Asegúrate de que el archivo JSON de la clave esté presente en la ruta especificada por `EE_PRIVATE_KEY_PATH`.

3.  **Ejecutar el servicio:**

    ```bash
    python app.py
    ```

> El servicio se ejecutará en `http://localhost:5001`.

---

## 👨‍💻 Autor y Especialista

Este proyecto ha sido desarrollado por **Pedro Alcoba Gómez**.

Pedro es un técnico ambiental especializado en **Sistemas de Información Geográfica (SIG)**, teledetección y desarrollo web orientado a cartografía interactiva, combinando trabajo de campo con tecnologías geoespaciales avanzadas como Google Earth Engine.

### 🔗 Enlaces y Contacto

- **Web Personal/Proyectos:** [pedralcg.github.io](https://pedralcg.github.io)
- **LinkedIn:** [linkedin.com/in/pedro-alcoba-gomez](https://linkedin.com/in/pedro-alcoba-gomez)
- **Email:** pedralcg.dev@gmail.com

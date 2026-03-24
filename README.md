# Experimentos-Energy-Neuronals-Networks

# Experimental Deep Learning: ADMM Energy-Budget Training

Este repositorio contiene experimentos y previsualizaciones gráficas relacionados con el entrenamiento de arquitecturas de Deep Learning bajo **restricciones de energía** utilizando el Método de Multiplicadores de Dirección Alterna (**ADMM**).

El tema central de los experimentos radica en la imposición del método **HARD**, el cual distribuye un presupuesto de energía máximo ($\tau$) a las distintas capas métricas de una red neuronal (Layers). A través de la **proyección sobre la norma nuclear** iterativamente durante el entrenamiento, obligamos a las matrices de pesos a reducir su Rango Efectivo (Effective Rank) y mantener una alta eficiencia y compresión sin dañar letalmente la precisión predictiva (Accuracy).

## 🗂 Estructura del Proyecto

*   **`import torch.py`**: A pesar de su nombre exótico, es un script en Python independiente que utiliza librerías tradicionales (`numpy`, `matplotlib`) para generar visualizaciones académicas estéticas simulando los resultados teóricos de este método. Se usa para crear figuras limpias sobre curvas de pérdida, energía y estabilidad asintótica para exportarlas a *Papers* (Figura 1, Figura 2 y Figura 4).
*   **`Paper/EXPARTFINAL.ipynb`**: Es el **experimento real**. Consiste en un *Jupyter Notebook* donde la red neuronal (Perceptrones Multicapa) se construye, instanciada mediante PyTorch. 
    *   Ejecuta entrenamientos y validaciones reales sobre un dataset (MNIST).
    *   Compara el entrenamiento restringido **HARD** frente a entrenamiento estándar sin restricciones (**STD**) y regularización base (**WD**).
    *   Realiza un análisis algorítmico profundo calculando matricialmente la "reducción" en complejidad en cada una de sus profundidades paramétricas (*Layer 1*, *Layer 2*...).

## 🚀 Requisitos e Instalación

### Experimento Principal (Jupyter Notebook)
Para poder abrir y ejecutar las matemáticas del notebook, asegúrate de tener entornos compatibles instalados:
1.  Es necesario Python 3 y Jupyter Notebook o Lab.
2.  Instala las librerías base para Deep Learning si usas el método local:
    ```bash
    pip install torch torchvision numpy matplotlib
    ```
3.  Ingresa a la carpeta correspondiente e inicia tu entorno:
    ```bash
    cd Paper/
    jupyter notebook EXPARTFINAL.ipynb
    ```

### Script Visual
Para crear de inmediato las métricas gráficas estéticas orientadas al documento (Figuras sintéticas):
1.  Ubicado en la raíz del proyecto, asegúrate de correrlo desde terminal en un entorno gráfico adecuado:
    ```bash
    python "import torch.py"
    ```
2.  Al cerrar cada ventana interactiva de Matplotlib en tu pantalla, verás que el progreso continúa y los archivos de imagen en alta fidelidad (`.png`) se irán guardando solos en el mismo entorno de trabajo.

## 📊 Metas de la Metodología 

Esta aproximación arquitectónica no requiere comprimir los modelos matemáticos a posteriori (*post-training pruning* o *quantization*), previniendo activamente gastos de sobre-parametrización descontrolada con garantías teóricas.

Esta metodología ha sido evaluada comprobándose qué tan robustamente los modelos mantienen un margen de adaptabilidad frente a configuraciones de **Profundidad / Effect of Depth** considerables (arquitecturas masivas desde 3 y extendiéndose hasta 9 capas paramétricas completas), mientras su frontera subyacente de energía jamás perfora el límite tolerado por diseño ($\tau$).

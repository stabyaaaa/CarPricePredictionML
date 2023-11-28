FROM python:3.9-bookworm
    RUN pip3 install ipykernel
    # RUN pip3 install dash
    # RUN pip3 install dash
    RUN pip3 install flask
    RUN pip3 install pandas 
    RUN pip3 install seaborn
    RUN pip3 install ppscore
    RUN pip3 install shap
    RUN pip3 install numpy==1.23.0
    RUN pip3 install mlflow
    RUN pip3 install scikit-learn
    RUN pip3 install tensorflow
    CMD tail -f /dev/null

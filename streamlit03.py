import streamlit as st
import pandas as pd
import numpy as np
from sympy.solvers import solve
import sympy as operador
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import altair as alt
import plotly.express as px

 

st.title('Biología del desarrollo')
st.header("Morfogénesis")
st.write("Tanto los factores bióticos como físicos, incluida la temperatura y diversas señales moleculares que emanan de patógenos, comensales y organismos específicos, afectan los resultados del desarrollo.")
st.write("Las decisiones del diseño anatómico se ven fuertemente afectadas por las señales laterales que se originan desde el exterior del genoma cigótico.")
st.write("Las vias endógenas dirigidas por estas influencia a menudo muestran efectos transgeneraciones, lo que les permite dar forma a la evolución de las anatomías incluso más rápido que la asimilación trdicional del tipo Baldwin.")
st.header("Serie de Fibonacci")
st.write("Se concoe como secuencia de Fibonacci a aquella definida por la relación de recurrencia:")
st.write(r"""
        La sucesión de Fibonacci sigue la ecuación de recurrencia lineal homogénea de orden 2
        $a_{n+2}= a_{n+1}+ a_{n}$
        """)
st.write("para los dos primeros térmnos de la serie:")
st.write(r"""
        $a_{0}=0;   a_{1}= 1$
        """)
st.write("Los primeros términos de la serie de Fibonacci serán:")

st.write(r""" 
        \{0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, ...\}
        """)
st.write(r"""
        Para determinar el término general de lase seuencia de fibonacci vamos a emplear el concepto matemático 'Operador', para ello vamos a describir inicialmnente el operador D muy utilizado en encuaciones diferenciales ordinarias EDO.
        Un operador $D$ es un elemento matemático que tomando una función devuelve otra, como por ejemplo. 
        """)
st.write(r"""
        Sea la función $y=f(x)$, aplicando el operador $D$ nos devuelve otra función como por ejemplo: $y'= D^{1};y'= (\frac{df}{dx})$.
        """)
st.write(r"""
        Si aplicamos el operador $D$, sucesivas veces, como por ejejmplo: $y''= D(Dy)= D^{2}y$.
        Para obtener el término general de nuestra serie Fubonacci creamos el operador $F$ de modo que el término $a_{n+1}= F a_{n}$
        """)
st.write(r"""
        - $a_{n+2}= F a_{n+1}$ ; $a_{n+1}= F a_{n}$
        - si aplicamos nuestro operador de modo sucesico, tendremos:
        - $a_{n+2}= F(F a_{n+1})= F^{2} a_{n}$
        - Generalizando:
        - $a_n= F^{n} a_{0}$
        """)
st.write(r"""
        La serie de Fibonacci tal y como ya hemos descrito:  
        - $a_{n+2}= a_{n+1}+ a_{n}$ 
        - Aplicando el operador F, tendremos:
        - $F^2a_{n}= Fa_{n}+ a_{n}$
        - $F^2a_{n}- Fa_{n}- a_{n}= 0$ 
        - $(F^2- F- 1)a_n= 0$
        - Para valores de $a_n$ distinto de cero, calculamos las raices de: $(F^2- F- 1)$
        """)
if st.checkbox('Mostrar raices calculadas de: (F²- F- 1)'):
    F= operador.symbols('F')
    exp= (F**2 -F -1)
    A= solve(exp)
    st.write(pd.DataFrame({
        'A': A
    }))

st.write(r"""
        Aquí se inicia el nuevo cálculo
        """)

if st.checkbox('Mostrar raices calculadas de: (dy/dn)'):
    x= operador.symbols('x')
    exp= (1/(5**(1/2)))* x* ((((1-5**(1/2))/2)/ ((1+5**(1/2))/2))**(x-1) -1)
    x= solve(exp)
    st.write(pd.DataFrame({
        'x': x
    }))
    n= np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16, 17, 18, 19, 20])
    a_n= (1/(5**(1/2)))* n* ((((1-5**(1/2))/2)/ ((1+5**(1/2))/2))**(n-1) -1)
    fibonacci= pd.DataFrame({
            'n': n[2:],
            'a_n': 1/a_n[2:]})
    
    if st.checkbox('Mostrar tabla de la derivada de la serie de Fibonacci y gráfico scatter'):
        st.subheader('Serie de la derivada de Fibonacci por filas')
        st.write(fibonacci)
        #st.line_chart(fibonacci)  # representa una curva line
        fig = px.scatter(
            fibonacci,
            x='n',
            y='a_n',
        )
        st.plotly_chart(fig)

st.write(r"""
        Aquí se termina
        """)    

st.write(r"""
        - Expresando las soluciones de forma racional, tendremos que el opeardor $F$ será: $\frac{1}{2}\pm \frac{\sqrt{5}}{2}$
        """)

st.write(r"""
        La solución geneal a las dos que hemos obtendio será una combinción lineal de estas, para ello debemos tomar dos valore de la secuencia.
        - $a_n= {(\frac{1}{2}- \frac{\sqrt{5}}{2})}^{n} A+ {(\frac{1}{2}+ \frac{\sqrt{5}}{2})}^{n} B$
        - Para los dos primeros valores de la serie:
        - $a_0=0$  
        - $a_1=1$
        """)

st.write(r"""
        Con los dos primerero valores de la serie se obtiene el sistema de ecuaciones que a continuación se muestra.
        """)
A = operador.symbols('A')
B = operador.symbols('B')
indice= np.array([0, 1, 2, 3, 4, 5 , 6, 7, 8])
serie0= np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])       
exp0= []
exp1= []
for n in indice:
    a_n= serie0[n]
    a_m= serie0[n+1] 
    e0 = ([((1+ (5**(1/2)))/2)**n*A+ ((1- (5**(1/2)))/2)**n*B- a_n, ((1+ (5**(1/2)))/2)**(n+1)*A+ ((1- (5**(1/2)))/2)**(n+1)*B- a_m])
    exp0.append(e0)
    exp= solve(exp0[-1])

st.table(pd.DataFrame({
    'n': indice,
    'n+1': indice+ 1,
    'expresión(n)': exp0
}))

st.write(r"""
        Los valores  que dan solución al sistema de ecuaciones para A y B son:
        """)

sol= solve(e0)
st.write(pd.DataFrame({
    'A': [exp[A]],
    'B': [exp[B]],
}))

st.write(r"""
        - La solución obtenida se puede expresar igualmente como $A= -B=\frac{1}{\sqrt{5}}= 0.44721...$
        - $a_n= {(\frac{1}{2}- \frac{\sqrt{5}}{2})}^{n} \frac{1}{\sqrt{5}}+ {(\frac{1}{2}+ \frac{\sqrt{5}}{2})}^{n} (-\frac{1}{\sqrt{5}})$
        - $a_n= \frac{1}{\sqrt{5}} \left(\left(\frac{1-\sqrt{5}}{2}\right)^n- \left(\frac{1+\sqrt{5}}{2}\right)^n\right)$
        """)

if st.checkbox('Mostrar tabla de la serie de Fibonacci aplicando el modelo obtenido'):
    n= np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16, 17, 18, 19, 20])
    a_n= ( 1/5**(1/2)  ) * ( ( ( 1+ 5**(1/2) ) /2)**n - ( (1- 5**(1/2) )/2 )**n )
    fibonacci= pd.DataFrame({
            'n': n,
            'a_n': a_n})
    st.subheader('Serie de Fibonacci por filas')
    st.write(fibonacci)
    


@st.cache
def load_data():
    fichero= "./datos.xlsm"
    data = pd.read_excel(fichero, sheet_name= "hoja01", skiprows=0, header=[0], index_col=0)
    return data

data_load_state = st.text('Loading data...')
data = load_data()
#print (data)
data_load_state.text('Loading data... done!')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

    st.subheader('Number of pickups by hour')
    st.bar_chart(data.value)

n = st.sidebar.slider('n', 1, 1500, 900)
c = st.sidebar.slider('c', 0, 10, 1)
phi = st.sidebar.slider('phi', 135.0, 157.5, 137.5)

n_imagen = st.sidebar.slider('n_imagen', 1, 5, 1)

i = np.arange(n)
r = c* np.sqrt(i)
theta= i*phi* ( np.pi / 180.0 )

x = r * np.cos(theta)
y = r * np.sin(theta)

fibonacci= pd.DataFrame({
        'x': x,
        'y': y})

f_img= "ImagenAA_00_"+ str(int(n_imagen)) +".png"
st.image(f_img, width= 300, caption= 'flor')

fig = px.scatter(
    fibonacci,
    x='x',
    y='y',
)
fig.update_layout(width=900,height=900)

st.plotly_chart(fig)

st.header('Fibonacci')






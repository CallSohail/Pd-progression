"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st


class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self, st_btn_select=None):

        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        st.write("""<style>font-size:100px !important;</style>""", unsafe_allow_html=True)
        st.markdown(
            """<style>
        .boxBorder1 {
            outline-offset: 5px;
            font-size:20px;
        }</style>
        """, unsafe_allow_html=True)
        from st_btn_select import st_btn_select

        app = st_btn_select(
            # The different pages
            self.apps,
            # Enable navbar
            # nav=True,
            # You can pass a formatting function. Here we capitalize the options
            format_func=lambda app: '{}'.format(app['title']),
        )


        app['function']()

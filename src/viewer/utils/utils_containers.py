import streamlit as st

def panel_manage_containers():
    if st.session_state.app_type == "cloud":
        st.error("This functionality isn't available on NiChart Cloud.")
        return
    try:
        import docker
        client = docker.from_env()
        client.ping()

        st.success("Connected to the local Docker daemon service.")
        images = client.images.list()
        if not images:
            st.info("No container images found. New images will be pulled as you run pipelines.")
        else:
            col1, col2, col3, col4 =st.columns([4, 2, 2, 2])
            col1.markdown("**Image Tags**")
            col2.markdown("**Image ID**")
            col3.markdown("**Size (MB)**")
            col4.markdown("**Action**")

            for idx, image in enumerate(images):
                tags = ", ".join(image.tags) if image.tags else "<untagged>"
                size_mb = round(image.attrs['Size'] / (1024 ** 2), 2)
                short_id = image.short_id

                col1, col2, col3, col4 = st.columns([4, 2, 2, 2])
                col1.write(tags)
                col2.code(short_id, language="text")
                col3.write(f"{size_mb}")
                if col4.button("Delete", key=f"delete_{idx}"):
                    try:
                        client.images.remove(image.id, force=True)
                        st.success(f"Deleted image {tags}")
                    except Exception as e:
                        st.error(f"Failed to delete image: {e}")
    except Exception as e:
        st.error(f"Could not connect to Docker daemon: {e}")
        

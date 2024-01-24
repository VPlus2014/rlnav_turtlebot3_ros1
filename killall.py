def main():
    import envs

    envs.bash.killall(
        process=[
            "rosmaster",
            "rosout",
            "roslaunch",
            "gzclient",
            "gzserver",
            "python",
            "python3",
        ],
    )


if __name__ == "__main__":
    main()

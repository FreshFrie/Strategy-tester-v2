from variance_phase_tracker import main

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('\nException:', e)
